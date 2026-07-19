"""Find LLM servers on the local network.

Typing an IP address in by hand assumes the reader knows it. Usually they
don't: they know they turned LM Studio on in the study. Nothing announces
itself on the network -- neither LM Studio nor Ollama has a discovery
protocol, no mDNS, no broadcast -- so the only way to find one is to go and
knock on every door.

That's what this does: a TCP connect to each address on this machine's own
/24, on the ports these servers actually use, and then a real /models request
to whatever answered. The models request is the part that matters. An open
port proves something is listening, not that it can analyse a book, and a
list of hopefuls that turn out to be printers is worse than no list at all.
Only servers that name at least one chat model come back.

Every port on every machine is tried, and each server found is its own entry:
one computer often runs LM Studio and Ollama side by side, on different ports,
neither aware of the other. To us they are two workers with two model lists.
"""

import ipaddress
import socket
import threading

from audiobook import analyze

# Where these servers live if nobody moved them: LM Studio, Ollama, then
# llama.cpp/vLLM, which speak enough of the same API to be worth a knock.
#
# All of them, on every machine. One computer commonly runs more than one -
# LM Studio and Ollama sit on different ports and don't know about each other
# - and they're separate workers to us, each with its own models. Stopping at
# the first one found on a host would silently halve that machine.
COMMON_PORTS = (1234, 11434, 8080, 8000)

_CONNECT_TIMEOUT = 0.30     # a machine on your own LAN answers in ~1ms
_MODELS_TIMEOUT = 2.0
_WORKERS = 96


def local_ipv4s():
    """This machine's IPv4 addresses on real networks.

    The UDP connect is a trick, not a conversation: no packet is sent, but
    the OS has to pick the route it *would* use, which is how you find the
    address that faces the network the reader means. gethostbyname_ex alone
    tends to answer 127.0.0.1 on Windows, or miss a second adapter.
    """
    out = []
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        out.append(s.getsockname()[0])
    except Exception:
        pass
    finally:
        s.close()
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in out:
                out.append(ip)
    except Exception:
        pass
    return [ip for ip in out if not ip.startswith("127.")]


def subnets(ips=None):
    """The /24 each of our addresses sits on, as [(network, our_ip)].

    /24 because that is what a house or an office is, and because the honest
    alternative is unbounded: a /16 is 65k addresses, and a scan that takes
    ten minutes is one nobody waits for.
    """
    out = []
    for ip in (ips if ips is not None else local_ipv4s()):
        try:
            net = ipaddress.ip_network(ip + "/24", strict=False)
        except ValueError:
            continue
        if not any(net == n for n, _ in out):
            out.append((net, ip))
    return out


def _port_open(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(_CONNECT_TIMEOUT)
    try:
        return s.connect_ex((host, port)) == 0
    except Exception:
        return False
    finally:
        s.close()


def scan(ports=(), skip=(), on_progress=None, should_stop=None):
    """Knock on every address of our /24s. Returns [{"url", "model", ...}].

    ports: try these first; the usual suspects are always included too.
    skip:  server URLs already configured, so they aren't offered twice.
    """
    want = []
    for p in list(ports) + list(COMMON_PORTS):
        try:
            p = int(p)
        except (TypeError, ValueError):
            continue
        if 0 < p < 65536 and p not in want:
            want.append(p)

    # by whole URL, not by host: one machine can run LM Studio and Ollama on
    # different ports, and having added one is no reason to hide the other
    skip_urls = set()
    for u in skip or ():
        u = (u or "").strip().rstrip("/").lower()
        if u:
            skip_urls.add(u)

    mine = local_ipv4s()
    targets = []
    for net, _ip in subnets(mine):
        for addr in net.hosts():
            targets.append(str(addr))
    # our own machine is already in the list as "This computer"
    targets = [t for t in targets if t not in mine]

    total = len(targets) * len(want)
    if not total:
        return []

    stopped = (lambda: False) if should_stop is None else should_stop
    lock = threading.Lock()
    found, done, idx = [], [0], [0]

    def work():
        while True:
            if stopped():
                return
            with lock:
                i = idx[0]
                idx[0] += 1
            if i >= len(targets):
                return
            host = targets[i]
            for port in want:
                if stopped():
                    return
                open_ = _port_open(host, port)
                with lock:
                    done[0] += 1
                    if on_progress:
                        on_progress(done[0], total)
                if not open_:
                    continue
                url = f"http://{host}:{port}"
                try:
                    models = analyze.list_models(url, timeout=_MODELS_TIMEOUT)
                except Exception:
                    models = []
                if not models:
                    continue        # listening, but not an LLM server
                loaded = [m["id"] for m in models if m["loaded"]]
                with lock:
                    found.append({
                        "url": url,
                        "model": (loaded or [m["id"] for m in models])[0],
                        "models": len(models),
                        "kind": models[0].get("kind", ""),
                        "known": url.lower() in skip_urls,
                    })
                # no break: keep going through the ports. A machine running
                # both LM Studio and Ollama is two workers, not one.

    threads = [threading.Thread(target=work, daemon=True, name=f"ab-scan{i}")
               for i in range(min(_WORKERS, len(targets)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    def order(e):
        host, _, port = e["url"].split("//")[1].partition(":")
        return tuple(int(x) for x in host.split(".")) + (int(port or 80),)

    found.sort(key=order)
    return found
