# Read macOS IOReport counters without sudo (GPU performance-state residency, energy, memory-controller counters).
#   ioreport.py list [filter]                      every (group, subgroup) with channel names
#   ioreport.py sample SECONDS INTERVAL G[/SG] ..   deltas per interval; state channels print residency shares
# Read-only: it subscribes to kernel counters, it changes nothing.
import ctypes as C, sys, time
cf = C.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
io = C.CDLL("/usr/lib/libIOReport.dylib")
vp = C.c_void_p
def sig(lib, name, res, args):
    f = getattr(lib, name); f.restype = res; f.argtypes = args; return f
CFStr = sig(cf, "CFStringCreateWithCString", vp, [vp, C.c_char_p, C.c_uint32])
CFGet = sig(cf, "CFStringGetCString", C.c_bool, [vp, C.c_char_p, C.c_long, C.c_uint32])
ACount = sig(cf, "CFArrayGetCount", C.c_long, [vp]); AGet = sig(cf, "CFArrayGetValueAtIndex", vp, [vp, C.c_long])
DGet = sig(cf, "CFDictionaryGetValue", vp, [vp, vp]); DCopy = sig(cf, "CFDictionaryCreateMutableCopy", vp, [vp, C.c_long, vp])
Rel = sig(cf, "CFRelease", None, [vp])
All = sig(io, "IOReportCopyAllChannels", vp, [C.c_uint64, C.c_uint64])
InGroup = sig(io, "IOReportCopyChannelsInGroup", vp, [vp, vp, C.c_uint64, C.c_uint64, C.c_uint64])
Merge = sig(io, "IOReportMergeChannels", None, [vp, vp, vp])
Sub = sig(io, "IOReportCreateSubscription", vp, [vp, vp, C.POINTER(vp), C.c_uint64, vp])
Samp = sig(io, "IOReportCreateSamples", vp, [vp, vp, vp]); Delta = sig(io, "IOReportCreateSamplesDelta", vp, [vp, vp, vp])
gG = sig(io, "IOReportChannelGetGroup", vp, [vp]); gS = sig(io, "IOReportChannelGetSubGroup", vp, [vp])
gN = sig(io, "IOReportChannelGetChannelName", vp, [vp]); gF = sig(io, "IOReportChannelGetFormat", C.c_int, [vp])
gU = sig(io, "IOReportChannelGetUnitLabel", vp, [vp]) if hasattr(io, "IOReportChannelGetUnitLabel") else None
sI = sig(io, "IOReportSimpleGetIntegerValue", C.c_int64, [vp, C.c_int])
stC = sig(io, "IOReportStateGetCount", C.c_int, [vp]); stN = sig(io, "IOReportStateGetNameForIndex", vp, [vp, C.c_int])
stR = sig(io, "IOReportStateGetResidency", C.c_int64, [vp, C.c_int])
U8 = 0x08000100
def S(s): return CFStr(None, s.encode(), U8)
def P(r):
    if not r: return ""
    b = C.create_string_buffer(512)
    return b.value.decode(errors="replace") if CFGet(r, b, 512, U8) else ""
KEY = S("IOReportChannels")
def chans(d):
    a = DGet(d, KEY)
    return [AGet(a, i) for i in range(ACount(a))] if a else []

if sys.argv[1] == "list":
    flt = sys.argv[2].lower() if len(sys.argv) > 2 else ""
    groups = {}
    for ch in chans(All(0, 0)):
        k = (P(gG(ch)), P(gS(ch)), gF(ch)); groups.setdefault(k, []).append(P(gN(ch)))
    for (g, s, f), names in sorted(groups.items()):
        if flt and flt not in (g + "/" + s + " " + " ".join(names)).lower(): continue
        print(f"{g} / {s}  fmt={f} n={len(names)}: " + ", ".join(names[:int(sys.argv[3]) if len(sys.argv) > 3 else 10]))
    sys.exit()

dur, step = float(sys.argv[2]), float(sys.argv[3])
want = None
for spec in sys.argv[4:]:
    g, _, s = spec.partition("/")
    d = InGroup(S(g), S(s) if s else None, 0, 0, 0)
    if not d: print("no such group", spec); continue
    if want is None: want = DCopy(None, 0, d)
    else: Merge(want, d, None)
subbed = vp(); sub = Sub(None, want, C.byref(subbed), 0, None)
only = None
import os
if os.environ.get("ONLY"): only = [x.lower() for x in os.environ["ONLY"].split(",")]
MINSHARE = float(os.environ.get("MINSHARE", "0.01"))
prev, t0 = Samp(sub, subbed, None), time.time(); tp = t0
while time.time() - t0 < dur:
    time.sleep(step)
    cur, tc = Samp(sub, subbed, None), time.time(); d = Delta(prev, cur, None); dt = tc - tp
    out = []
    for ch in chans(d):
        name = P(gN(ch)); full = P(gG(ch)) + "/" + P(gS(ch)) + "/" + name
        if only and not any(o in full.lower() for o in only): continue
        if gF(ch) == 2:
            n = stC(ch); res = [(P(stN(ch, i)), stR(ch, i)) for i in range(n)]; tot = sum(r for _, r in res) or 1
            out.append(P(gS(ch)) + ":" + name + "{" + " ".join(f"{k.strip()}:{r / tot:.3f}" for k, r in res if r / tot >= MINSHARE) + "}")
        elif gF(ch) == 1:
            v = sI(ch, 0)
            if v: out.append(f"{name}={v / dt:.4g}/s" + (f"[{P(gU(ch))}]" if gU else ""))
    print(f"{tc:.3f} dt={dt:.3f} " + " | ".join(out), flush=True)
    Rel(prev); Rel(d); prev, tp = cur, tc
