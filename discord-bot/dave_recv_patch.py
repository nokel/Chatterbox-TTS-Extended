# dave_recv_patch.py - teach discord-ext-voice-recv to decrypt DAVE (E2EE) audio.
#
# Since March 2026 Discord requires the DAVE end-to-end-encryption protocol on
# voice connections. discord.py 2.7 implements DAVE for *sending* audio, but
# the voice-receive extension (0.5.2a179, June 2025) predates enforcement:
# after transport decryption the incoming frames are still E2EE, so the opus
# decoder gets ciphertext and raises "corrupted stream", killing the packet
# router. This module:
#   1. wraps the reader callback to run davey frame decryption per packet
#      (skipped for passthrough/transition frames, which arrive unencrypted);
#   2. makes the packet decoder emit silence instead of dying when a frame
#      still cannot be decoded.
#
# Written against discord.py 2.7.1 + discord-ext-voice-recv 0.5.2a179.

import logging

import davey
from discord.opus import OpusError
from discord.ext.voice_recv import opus as _vr_opus
from discord.ext.voice_recv import reader as _reader
from discord.ext.voice_recv import rtp as _rtp
from nacl.exceptions import CryptoError

log = logging.getLogger(__name__)

_SILENCE_PCM = b"\x00" * _vr_opus.Decoder.FRAME_SIZE if hasattr(_vr_opus.Decoder, "FRAME_SIZE") else b"\x00" * 3840


_diag_counts: dict = {}


def _diag(key, msg, *args):
    n = _diag_counts.get(key, 0)
    if n < 3:
        _diag_counts[key] = n + 1
        log.warning(msg, *args)


def _dave_frame(reader, packet) -> bytes:
    """E2EE-decrypt one RTP frame when a DAVE session is active."""
    data = packet.decrypted_data
    if not data:
        return data
    conn = reader.voice_client._connection
    sess = getattr(conn, "dave_session", None)
    if sess is None or not getattr(conn, "dave_protocol_version", 0):
        _diag("nosess", "DAVE: no session (ver=%s), passing frame through",
              getattr(conn, "dave_protocol_version", None))
        return data
    user_id = reader.voice_client._ssrc_to_id.get(packet.ssrc)
    if user_id is None:
        _diag(("nouser", packet.ssrc),
              "DAVE: no user mapping for ssrc %s yet", packet.ssrc)
        return data
    try:
        return sess.decrypt(int(user_id), davey.MediaType.audio, data)
    except Exception as e:
        # Transition/passthrough frames arrive unencrypted; use them as-is.
        _diag(("fail", user_id),
              "DAVE decrypt failed for user %s (ready=%s, users=%s): %r",
              user_id, sess.ready, sess.get_user_ids(), e)
        return data


def _patched_callback(self, packet_data: bytes) -> None:
    # Faithful copy of AudioReader.callback with one added DAVE decrypt step.
    packet = rtp_packet = rtcp_packet = None
    try:
        if not _rtp.is_rtcp(packet_data):
            packet = rtp_packet = _rtp.decode_rtp(packet_data)
            packet.decrypted_data = self.decryptor.decrypt_rtp(packet)
            packet.decrypted_data = _dave_frame(self, packet)  # <-- added
        else:
            packet = rtcp_packet = _rtp.decode_rtcp(self.decryptor.decrypt_rtcp(packet_data))
            if not isinstance(packet, _rtp.ReceiverReportPacket):
                log.debug("Received unexpected rtcp packet: type=%s, %s", packet.type, type(packet))
    except CryptoError:
        log.error("CryptoError decoding packet data")
        return
    except Exception:
        if self._is_ip_discovery_packet(packet_data):
            return
        log.exception("Error unpacking packet")
    finally:
        if self.error:
            self.stop()
            return
        if not packet:
            return

    if rtcp_packet:
        self.packet_router.feed_rtcp(rtcp_packet)
    elif rtp_packet:
        ssrc = rtp_packet.ssrc
        if ssrc not in self.voice_client._ssrc_to_id:
            if rtp_packet.is_silence():
                return
            else:
                log.info("Received packet for unknown ssrc %s", ssrc)
        self.speaking_timer.notify(ssrc)
        try:
            self.packet_router.feed_rtp(rtp_packet)
        except Exception as e:
            log.exception("Error processing rtp packet")
            self.error = e
            self.stop()


_orig_decode_packet = _vr_opus.PacketDecoder._decode_packet


def _safe_decode_packet(self, packet):
    try:
        return _orig_decode_packet(self, packet)
    except OpusError:
        log.warning("Undecodable frame from ssrc %s; emitting silence", self.ssrc)
        return packet, _SILENCE_PCM


def apply() -> None:
    _reader.AudioReader.callback = _patched_callback
    _vr_opus.PacketDecoder._decode_packet = _safe_decode_packet
    log.info("voice-recv DAVE patch applied (davey protocol v%s)",
             davey.DAVE_PROTOCOL_VERSION)
