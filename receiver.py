# rx.py
import numpy as np
import sounddevice as sd
import scipy.signal as sig
import crcmod.predefined
import queue
import threading

fs             = 48000      # sample rate
Baud           = 100        # symbols per second
f0, f1          = 300, 1000  # mark (0) & space (1) freqs in Hz
THRESH_DELTA   = 0          # delta > 0 ⇒ bit=1; delta <=0 ⇒ bit=0
PREAMBLE_BITS  = [1,1,1,0,0,0,1,0,0,1,0]
PAYLOAD_LENGTH = 7         # bytes of payload you expect
rx_device      = None       # set to None to auto‑select default mic

Tb       = int(fs / Baud)   # samples per symbol
crc16    = crcmod.predefined.mkCrcFun('x25')
audio_q  = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """Called in real time by sounddevice for each block of Tb samples."""
    if status:
        print("Stream status:", status)
    chunk = indata[:,0].copy()
    # RMS logging
    rms = np.sqrt(np.mean(chunk**2))
    # print(f"RMS={rms:.4f}", end=' | ')
    audio_q.put(chunk)

def start_stream():
    """Start the input stream and return it."""
    stream = sd.InputStream(
        samplerate=fs,
        blocksize=Tb,
        device=rx_device,
        channels=1,
        callback=audio_callback
    )
    stream.start()
    print("Recording started on device:", stream.device)
    return stream

def goertzel_block(x, f):
    k    = int(0.5 + Tb * f / fs)
    w    = 2 * np.pi * k / Tb
    coef = 2 * np.cos(w)
    s0 = s1 = 0.0
    for sample in x:
        s0, s1 = sample + coef*s0 - s1, s0
    return s0*s0 + s1*s1 - coef*s0*s1

def decoder():
    buf = np.empty(0, dtype=np.float32)
    while True:
        buf = np.concatenate((buf, audio_q.get()))
        while len(buf) >= Tb:
            sym, buf = buf[:Tb], buf[Tb:]
            p0 = goertzel_block(sym, f0)
            p1 = goertzel_block(sym, f1)
            delta = p1 - p0
            bit = 1 if delta > THRESH_DELTA else 0
            # print(f"p0={p0:.0f}, p1={p1:.0f}, Δ={delta:.0f} → {bit}")
            yield bit

def run_receiver():
    bit_gen = decoder()
    preamble = tuple(PREAMBLE_BITS)
    window = []

    while True:
        b = next(bit_gen)
        window.append(b)
        if len(window) > len(preamble):
            window.pop(0)

        if tuple(window) == preamble:
            print("\n+++ PREAMBLE DETECTED +++")
            total_bytes = 2 + PAYLOAD_LENGTH + 2
            total_bits  = total_bytes * 8
            frame_bits = []
            for _ in range(total_bits):
                frame_bits.append(next(bit_gen))
                print(frame_bits[-1], end="")
            data = np.packbits(frame_bits)

            # Convert header & CRC slices to bytes for comparison
            header = data[:2].tobytes()
            payload = data[2:2+PAYLOAD_LENGTH]
            crc_recv = int.from_bytes(data[-2:].tobytes(), 'big')

            print()
            if header != b'\x2A\x2A':
                print("Bad header:", header)
                continue

            crc_calc = crc16(payload.tobytes())
            if crc_calc == crc_recv:
                print("RX OK:", payload.tobytes().decode(errors='ignore'))
            else:
                print(f"CRC error: recv={crc_recv}, calc={crc_calc}")
                print(payload.tobytes().decode(errors='ignore'))

if __name__ == "__main__":
    stream = start_stream()
    try:
        run_receiver()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        stream.stop()
