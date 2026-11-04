#!/usr/bin/env python3
"""
Intercom Live : transcription en direct d'une com de régie, avec qui parle.

Deux modes :
  - mono  : un seul canal avec toute la com mixée (ex. sortie d'un Solidcom M1).
            Les interlocuteurs sont reconnus à la voix (empreinte vocale), « Voix 1 », « Voix 2 »…
            que tu renommes dans la fenêtre ; les noms sont retenus pour les émissions suivantes.
  - multi : un canal par poste (le nom vient du canal, aucune reconnaissance nécessaire).

Usage :
  python3 intercom_live.py --list-devices
  python3 intercom_live.py                      # utilise config.toml
  python3 intercom_live.py --file test.wav      # rejoue un fichier (test sans régie)
  python3 intercom_live.py --file test.wav --fake-asr   # teste la chaîne sans Whisper
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import queue
import re
import sys
import threading
import time
import tomllib
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ASR_RATE = 16000
PALETTE = ["#ff7043", "#29b6f6", "#66bb6a", "#ab47bc", "#ffca28", "#ec407a", "#26a69a",
           "#8d6e63", "#7e57c2", "#d4e157", "#5c6bc0", "#ffa726"]
VOICE_MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
                   "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")

# Phrases que Whisper « hallucine » souvent sur du silence ou du bruit, surtout en français.
HALLUCINATIONS = [
    r"sous-titrage", r"sous-titres? (réalisés|faits)", r"amara\.org", r"merci d'avoir regardé",
    r"abonnez-vous", r"n'oubliez pas de (vous abonner|liker)", r"à bientôt sur", r"st' ?501",
    r"radio-canada", r"^\W*(merci\.?|\.+|…)\W*$", r"^\W*$",
]
HALLU_RE = re.compile("|".join(HALLUCINATIONS), re.IGNORECASE)


# --------------------------------------------------------------------------- config

@dataclass
class Speaker:
    channel: int          # 1-based, comme sur l'interface audio
    name: str
    color: str = "#4fc3f7"
    gain_db: float = 0.0


@dataclass
class Config:
    mode: str = "mono"             # mono | multi
    device: str | int | None = None
    samplerate: int = 48000
    channel: int = 1               # canal utilisé en mode mono
    gain_db: float = 0.0
    speakers: list[Speaker] = field(default_factory=list)
    language: str = "fr"
    model: str = "large-v3-turbo"
    backend: str = "auto"          # auto | mlx | faster
    initial_prompt: str = ""
    vad_threshold_db: float = -45.0
    vad_hangover_ms: int = 700
    vad_preroll_ms: int = 300
    min_speech_ms: int = 350
    max_segment_s: float = 12.0
    voice_threshold: float = 0.40
    voice_model: str = "models/voix.onnx"
    voices_file: str = "voix.json"
    keywords: list[str] = field(default_factory=list)
    port: int = 8765
    log_dir: str = "transcripts"
    open_browser: bool = True
    update_repo: str = ""          # dépôt GitHub d'où viennent les mises à jour automatiques

    @classmethod
    def load(cls, path: Path) -> "Config":
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        audio, asr, vad, ui, voices = (raw.get(k, {}) for k in ("audio", "asr", "vad", "ui", "voix"))
        dev = audio.get("device")
        mode = audio.get("mode", "mono")
        return cls(
            mode=mode,
            device=None if dev in (None, "", "default") else dev,
            samplerate=int(audio.get("samplerate", 48000)),
            channel=int(audio.get("channel", 1)),
            gain_db=float(audio.get("gain_db", 0)),
            speakers=[Speaker(**s) for s in raw.get("speakers", [])],
            language=asr.get("language", "fr"),
            model=asr.get("model", "large-v3-turbo"),
            backend=asr.get("backend", "auto"),
            initial_prompt=asr.get("initial_prompt", ""),
            vad_threshold_db=float(vad.get("threshold_db", -45)),
            vad_hangover_ms=int(vad.get("hangover_ms", 450 if mode == "mono" else 700)),
            vad_preroll_ms=int(vad.get("preroll_ms", 300)),
            min_speech_ms=int(vad.get("min_speech_ms", 350)),
            max_segment_s=float(vad.get("max_segment_s", 12)),
            voice_threshold=float(voices.get("seuil", 0.40)),
            voice_model=voices.get("modele", "models/voix.onnx"),
            voices_file=voices.get("fichier", "voix.json"),
            keywords=list(ui.get("keywords", [])),
            port=int(ui.get("port", 8765)),
            log_dir=ui.get("log_dir", "transcripts"),
            open_browser=bool(ui.get("open_browser", True)),
            update_repo=str(raw.get("maj", {}).get("depot", "")),
        )


# Données de l'utilisateur (voix, réglages, transcriptions, modèle de voix) : hors du dossier de l'app,
# pour qu'une mise à jour (remplacer le dossier) ne les efface pas.
# (Pas dans Documents : macOS y bloque l'accès aux apps non signées lancées depuis le Dock.)
_SUPPORT = (Path.home() / "Library" / "Application Support" / "Intercom Live") if sys.platform == "darwin" \
    else (Path.home() / ".intercom-live")
DATA_DIR = Path(os.environ.get("INTERCOM_LIVE_DATA", _SUPPORT))


def _migrate_old_data():
    """Les anciennes versions rangeaient tout dans le dossier de l'app : on recopie une fois."""
    import shutil
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for rel in ("voix.json", "reglages.json", "models/voix.onnx"):
        dst = DATA_DIR / rel
        olds = (HERE, Path.home() / "Documents" / "Intercom Live", Path.home() / "Downloads" / "intercom-live")
        src = None
        for d in olds:
            try:
                if (d / rel).exists():
                    src = d / rel
                    break
            except OSError:   # dossier protégé par macOS
                pass
        if src and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"[données] {rel} déplacé vers {DATA_DIR}", flush=True)
    try:
        old_tr = Path.home() / "Documents" / "Intercom Live" / "transcripts"
        if old_tr.is_dir():
            (DATA_DIR / "transcripts").mkdir(exist_ok=True)
            for f in old_tr.glob("*.txt"):
                if not (DATA_DIR / "transcripts" / f.name).exists():
                    shutil.copy2(f, DATA_DIR / "transcripts" / f.name)
    except OSError:
        pass


def local_path(p: str) -> Path:
    return Path(p) if os.path.isabs(p) else DATA_DIR / p


SETTINGS_FILE = DATA_DIR / "reglages.json"   # choix faits dans la fenêtre (prioritaires sur config.toml)


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(**kw):
    d = load_settings()
    d.update(kw)
    SETTINGS_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- events (UI bus)

class EventBus:
    def __init__(self):
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self.history: list[dict] = []
        self.status = ""

    def set_status(self, text: str):
        self.status = text
        if text:
            print(f"[état] {text}", flush=True)
        self.publish({"type": "status", "text": text})

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, ev: dict):
        with self._lock:
            if ev["type"] == "line":
                self.history.append(ev)
                self.history = self.history[-500:]
            for q in self._subs:
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    pass


# --------------------------------------------------------------------------- segmentation

def resample(x: np.ndarray, sr_in: int, sr_out: int = ASR_RATE) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    k = sr_in / sr_out
    if abs(k - round(k)) < 1e-9 and k > 1:  # 48k -> 16k : moyenne (anti-repliement simple) + décimation
        k = int(round(k))
        n = len(x) // k * k
        return x[:n].reshape(-1, k).mean(axis=1).astype(np.float32)
    t_out = np.arange(int(len(x) / k)) * k
    return np.interp(t_out, np.arange(len(x)), x).astype(np.float32)


class ChannelSegmenter:
    """Découpe le flux d'un canal en prises de parole (VAD énergie avec hangover et pré-roll)."""

    FRAME_MS = 30

    def __init__(self, idx: int, speaker: Speaker, cfg: Config, sr: int, on_segment, on_state):
        self.idx, self.speaker, self.cfg, self.sr = idx, speaker, cfg, sr
        self.on_segment, self.on_state = on_segment, on_state
        self.frame = int(sr * self.FRAME_MS / 1000)
        self.gain = 10 ** (speaker.gain_db / 20)
        self.buf = np.zeros(0, np.float32)
        self.pre: list[np.ndarray] = []
        self.pre_n = max(1, cfg.vad_preroll_ms // self.FRAME_MS)
        self.hang_n = max(1, cfg.vad_hangover_ms // self.FRAME_MS)
        self.active = False
        self.silence = 0
        self.speech_frames = 0
        self.seg: list[np.ndarray] = []
        self.seg_t0 = 0.0
        self.level_db = -120.0
        self.noise_db = -40.0   # plancher de bruit estimé (souffle de la com, ventilation…)

    def feed(self, x: np.ndarray, t_now: float):
        self.buf = np.concatenate([self.buf, x * self.gain])
        while len(self.buf) >= self.frame:
            f, self.buf = self.buf[: self.frame], self.buf[self.frame:]
            rms = float(np.sqrt(np.mean(f * f)) + 1e-12)
            db = 20 * np.log10(rms)
            self.level_db = max(db, self.level_db - 1.5)  # décroissance pour le VU
            # plancher de bruit : descend vite, remonte lentement (~1,3 dB/s)
            self.noise_db = db if db < self.noise_db else self.noise_db + 0.04
            loud = db > max(self.cfg.vad_threshold_db, self.noise_db + 12)
            if not self.active:
                self.pre.append(f)
                if len(self.pre) > self.pre_n:
                    self.pre.pop(0)
                if loud:
                    self.active, self.silence, self.speech_frames = True, 0, 1
                    self.seg = list(self.pre)
                    self.pre = []
                    self.seg_t0 = t_now
                    self.on_state(self.idx, True)
            else:
                self.seg.append(f)
                if loud:
                    self.silence = 0
                    self.speech_frames += 1
                else:
                    self.silence += 1
                dur = len(self.seg) * self.FRAME_MS / 1000
                if self.silence >= self.hang_n:
                    self._flush(final=True)
                elif dur >= self.cfg.max_segment_s:
                    self._flush(final=False)

    def _flush(self, final: bool):
        if self.speech_frames * self.FRAME_MS >= self.cfg.min_speech_ms:
            audio = np.concatenate(self.seg)
            self.on_segment(self.idx, resample(audio, self.sr), self.seg_t0)
        if final:
            self.active = False
            self.on_state(self.idx, False)
            self.seg = []
        else:  # segment trop long : on coupe et on continue
            self.seg = []
            self.seg_t0 = time.time()
        self.speech_frames = 0
        self.silence = 0


# --------------------------------------------------------------------------- reconnaissance des voix

class VoiceBook:
    """Regroupe les prises de parole par voix (empreinte vocale + regroupement en ligne).

    Chaque voix garde un identifiant stable (index) pour que l'interface puisse la renommer,
    la fusionner, l'ignorer ou la supprimer. Une voix fusionnée garde son empreinte comme
    « variante » de la voix cible : la personne est mieux reconnue ensuite (micro différent,
    voix qui porte…). Les voix nommées ou ignorées sont enregistrées dans voix.json.
    """

    MIN_NEW_S = 1.0     # une prise plus courte ne peut pas créer une nouvelle voix

    def __init__(self, cfg: Config):
        import sherpa_onnx
        self.cfg = cfg
        model = local_path(cfg.voice_model)
        if not model.exists():
            model.parent.mkdir(parents=True, exist_ok=True)
            print("[voix] téléchargement du modèle d'empreinte vocale (~40 Mo)…", flush=True)
            urllib.request.urlretrieve(VOICE_MODEL_URL, model)
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=2))
        self.lock = threading.Lock()
        # {name, color, emb (np), n, named, alias (index cible si fusionnée), deleted, ignored}
        self.voices: list[dict] = []
        self.path = local_path(cfg.voices_file)
        self.next_n = 1
        self._load()

    @staticmethod
    def _new(name, color, emb, n=1, named=False, alias=None, ignored=False):
        return {"name": name, "color": color, "emb": emb, "n": n, "named": named, "alias": alias,
                "deleted": False, "ignored": ignored}

    # -- persistance
    def _load(self):
        if not self.path.exists():
            return
        for v in json.loads(self.path.read_text(encoding="utf-8")):
            embs = v.get("embs") or [v["emb"]]
            root = len(self.voices)
            self.voices.append(self._new(v["name"], v["color"], np.array(embs[0], np.float32), int(v.get("n", 5)),
                                         named=True, ignored=bool(v.get("ignored"))))
            for e in embs[1:]:   # variantes issues de fusions
                self.voices.append(self._new(v["name"], v["color"], np.array(e, np.float32), 5, True, alias=root))
        roots = [v for v in self.voices if v["alias"] is None]
        print(f"[voix] {len(roots)} voix connues : {', '.join(v['name'] + (' (ignorée)' if v['ignored'] else '') for v in roots)}",
              flush=True)

    def save(self):
        keep = []
        for i, v in enumerate(self.voices):
            if v["alias"] is not None or v["deleted"] or not (v["named"] or v["ignored"]):
                continue
            embs = [v["emb"]] + [w["emb"] for j, w in enumerate(self.voices)
                                 if j != i and not w["deleted"] and w["alias"] is not None and self._root(j) == i]
            keep.append({"name": v["name"], "color": v["color"], "n": v["n"], "ignored": v["ignored"],
                         "embs": [e.astype(float).round(5).tolist() for e in embs]})
        self.path.write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")

    # -- empreintes
    def embed(self, audio: np.ndarray) -> np.ndarray:
        s = self.ex.create_stream()
        s.accept_waveform(ASR_RATE, audio)
        s.input_finished()
        e = np.array(self.ex.compute(s), np.float32)
        return e / (np.linalg.norm(e) + 1e-9)

    def _root(self, i: int) -> int:
        seen = 0
        while self.voices[i]["alias"] is not None and seen < 100:
            i = self.voices[i]["alias"]
            seen += 1
        return i

    def _alive(self, i: int) -> bool:
        return not self.voices[i]["deleted"] and not self.voices[self._root(i)]["deleted"]

    def identify(self, audio: np.ndarray) -> tuple[int, bool]:
        """Renvoie (index de la voix « racine », nouvelle_voix?)."""
        dur = len(audio) / ASR_RATE
        e = self.embed(audio)
        with self.lock:
            best, sim = None, -1.0
            for i in range(len(self.voices)):      # toutes les empreintes, variantes comprises
                if not self._alive(i):
                    continue
                s = float(self.voices[i]["emb"] @ e)
                if s > sim:
                    best, sim = i, s
            if best is not None and (sim >= self.cfg.voice_threshold or dur < self.MIN_NEW_S):
                v = self.voices[best]
                w = min(v["n"], 30)  # la voix s'adapte doucement (fatigue, micro qui bouge…)
                m = v["emb"] * w + e
                v["emb"] = m / np.linalg.norm(m)
                v["n"] += 1
                return self._root(best), False
            while any(v["name"] == f"Voix {self.next_n}" for v in self.voices):
                self.next_n += 1
            self.voices.append(self._new(f"Voix {self.next_n}", PALETTE[len(self.voices) % len(PALETTE)], e))
            return len(self.voices) - 1, True

    def is_ignored(self, i: int) -> bool:
        with self.lock:
            return self.voices[self._root(i)]["ignored"]

    # -- actions depuis l'interface
    def _merge(self, src: int, dst: int):
        src, dst = self._root(src), self._root(dst)
        if src == dst:
            return
        self.voices[src]["alias"] = dst
        a = self.voices[dst]
        a["named"] = a["named"] or self.voices[src]["named"]
        a["n"] += self.voices[src]["n"]

    def act(self, i: int, action: str, name: str = "", into: int | None = None) -> None:
        with self.lock:
            i = self._root(i)
            v = self.voices[i]
            if action == "rename":
                name = name.strip()[:40]
                if not name:
                    return
                for j, w in enumerate(self.voices):  # nom déjà pris → fusion
                    if j != i and w["alias"] is None and not w["deleted"] and w["name"].lower() == name.lower():
                        self._merge(i, j)
                        break
                else:
                    v["name"], v["named"] = name, True
            elif action == "merge" and into is not None:
                self._merge(i, int(into))
            elif action == "delete":
                v["deleted"] = True
            elif action in ("ignore", "unignore"):
                v["ignored"] = action == "ignore"
            self.save()

    def public(self) -> list[dict]:
        with self.lock:
            out = []
            for i, v in enumerate(self.voices):
                r = self.voices[self._root(i)]
                out.append({"name": r["name"], "color": r["color"], "alias": v["alias"], "named": r["named"],
                            "deleted": v["deleted"] or r["deleted"], "ignored": r["ignored"], "n": v["n"]})
            return out


# --------------------------------------------------------------------------- ASR

class Transcriber:
    def __init__(self, cfg: Config, fake: bool = False):
        self.cfg = cfg
        self.backend = None
        if fake:
            self.backend = "fake"
            return
        want = cfg.backend
        if want in ("auto", "mlx"):
            try:
                import mlx_whisper  # noqa: F401
                self.backend = "mlx"
                self._mlx_repo = cfg.model if "/" in cfg.model else f"mlx-community/whisper-{cfg.model}"
            except ImportError:
                if want == "mlx":
                    raise
        if self.backend is None:
            from faster_whisper import WhisperModel
            print(f"[asr] chargement faster-whisper « {cfg.model} »…", flush=True)
            self._fw = WhisperModel(cfg.model, device="auto", compute_type="int8")
            self.backend = "faster"
        print(f"[asr] moteur : {self.backend}", flush=True)

    def warmup(self):
        if self.backend == "mlx":
            print(f"[asr] chargement mlx-whisper « {self._mlx_repo} » (premier lancement : téléchargement)…", flush=True)
            self.transcribe(np.zeros(ASR_RATE, np.float32))

    def transcribe(self, audio: np.ndarray) -> list[tuple[float, float, str]]:
        """Renvoie une liste de (début_s, fin_s, texte)."""
        dur = len(audio) / ASR_RATE
        if self.backend == "fake":
            time.sleep(0.05)
            self._n = getattr(self, "_n", -1) + 1
            demo = ["Attention, antenne dans trente secondes.", "Cam 2 tu resserres sur l'invité.",
                    "Le son, on n'a pas le HF de l'invité !", "Top VT dans 10, 9, 8…",
                    "OK c'est bon pour moi, on enchaîne sur le plateau."]
            return [(0.0, dur, f"{demo[self._n % len(demo)]} [{dur:.1f}s]")]
        prompt = self.cfg.initial_prompt or None
        if self.backend == "mlx":
            import mlx_whisper
            r = mlx_whisper.transcribe(
                audio, path_or_hf_repo=self._mlx_repo, language=self.cfg.language,
                initial_prompt=prompt, condition_on_previous_text=False, temperature=(0.0, 0.2, 0.4),
                compression_ratio_threshold=2.0, no_speech_threshold=0.6, logprob_threshold=-1.0,
            )
            return [(float(s["start"]), float(s["end"]), s["text"].strip())
                    for s in r.get("segments", []) if s.get("no_speech_prob", 0) < 0.6]
        segs, _ = self._fw.transcribe(
            audio, language=self.cfg.language, initial_prompt=prompt, beam_size=1,
            condition_on_previous_text=False, vad_filter=False, temperature=[0.0, 0.2, 0.4],
            compression_ratio_threshold=2.0,
        )
        return [(s.start, s.end, s.text.strip()) for s in segs if s.no_speech_prob < 0.6]


def collapse_repeats(t: str) -> str:
    """Whisper part parfois en boucle (« caméra 1, caméra 1, caméra 1… ») : on ne garde qu'une fois
    tout groupe de 1 à 8 mots répété 3 fois ou plus d'affilée (2 fois est gardé : « top, top »)."""
    words = t.split()
    key = [re.sub(r"[^\w]", "", w.lower()) for w in words]
    out, i = [], 0
    while i < len(words):
        done = False
        for n in range(1, 9):
            if i + 3 * n > len(words):
                break
            unit = key[i:i + n]
            if not any(unit):
                continue
            reps = 1
            while key[i + reps * n:i + (reps + 1) * n] == unit:
                reps += 1
            if reps >= 3:
                out.extend(words[i:i + n])
                i += reps * n
                done = True
                break
        if not done:
            out.append(words[i]); i += 1
    return re.sub(r"[,\s]+$", "", " ".join(out)) if len(out) < len(words) else t


def clean_text(t: str) -> str:
    t = collapse_repeats(re.sub(r"\s+", " ", t).strip())
    if not t or HALLU_RE.search(t):
        return ""
    return t


# --------------------------------------------------------------------------- alerte « ça me concerne »

DEFAULT_NAMES = ["Clément", "Clem"]
DEFAULT_WORDS = [
    "le son", "du son", "au son", "côté son", "ingé son", "régie son", "bande son", "son coupé", "pas de son",
    "micro", "HF", "retour", "niveau", "volume", "trop fort", "pas assez fort", "on n'entend", "on entend",
    "j'entends", "entend pas", "entend rien", "larsen", "souffle", "grésille", "sature", "saturé", "audio",
    "mixage", "console", "fader", "média", "playback", "musique", "jingle", "perche", "cravate", "oreillette",
    "ear", "casque", "PAD son", "virgule",
]


def _fold(t: str) -> tuple[str, list[int]]:
    """Minuscules sans accents + correspondance vers les positions du texte d'origine."""
    import unicodedata
    out, idx = [], []
    for i, ch in enumerate(t.replace("’", "'")):
        for c in unicodedata.normalize("NFD", ch):
            if unicodedata.category(c) != "Mn":
                out.append(c.lower())
                idx.append(i)
    return "".join(out), idx


class Detector:
    """Repère ce qui concerne l'ingé son : son prénom (même mal transcrit) et le vocabulaire du son."""

    def __init__(self, names: list[str], words: list[str], enabled: bool = True):
        import difflib
        self._ratio = lambda a, b: difflib.SequenceMatcher(None, a, b).ratio()
        self.enabled = enabled
        self.names = [n.strip() for n in names if n.strip()]
        self.words = [w.strip() for w in words if w.strip()]
        self._names = [_fold(n)[0] for n in self.names]
        pats = [r"\s+".join(re.escape(part) for part in _fold(w)[0].split()) + r"(?:s|x)?" for w in self.words]
        self._words_re = re.compile(r"(?<![\w])(" + "|".join(pats) + r")(?![\w])") if pats else None

    def find(self, text: str) -> list[str]:
        """Renvoie les passages du texte qui ont déclenché l'alerte (dans l'orthographe d'origine)."""
        if not self.enabled:
            return []
        folded, idx = _fold(text)
        spans = []
        for m in re.finditer(r"[a-z]+", folded):           # prénom, avec tolérance aux fautes de Whisper
            tok = m.group()
            for n in self._names:
                if tok == n or (len(n) >= 4 and len(tok) >= 4 and self._ratio(tok, n) >= 0.8):
                    spans.append((m.start(), m.end()))
                    break
        if self._words_re:
            spans += [(m.start(), m.end()) for m in self._words_re.finditer(folded)]
        out = []
        for a, b in sorted(spans):
            piece = text.replace("’", "'")[idx[a]: idx[b - 1] + 1]
            if piece and piece not in out:
                out.append(piece)
        return out


def load_detector(cfg: "Config") -> Detector:
    a = load_settings().get("alerte", {})
    names = list(a.get("prenoms", DEFAULT_NAMES))
    words = a.get("mots") or [k for k in DEFAULT_WORDS + list(cfg.keywords)
                              if _fold(k)[0] not in {_fold(n)[0] for n in names}]
    seen, uniq = set(), []
    for w in words:
        f = _fold(w)[0].strip()
        if f and f not in seen:
            seen.add(f); uniq.append(w)
    return Detector(names, uniq, a.get("actif", True))


GUI_ACTIVE = False
_flash_windows: list = []
_last_flash = 0.0


def flash_screen(force: bool = False):
    """Gros flash blanc sur tous les écrans, par-dessus toutes les apps (même en plein écran), et l'icône
    du Dock qui rebondit. Ne fait rien hors de la vraie fenêtre Mac (la page fait alors son propre flash)."""
    global _last_flash
    if not GUI_ACTIVE or (not force and time.time() - _last_flash < 3):
        return
    _last_flash = time.time()
    try:
        from PyObjCTools import AppHelper
        AppHelper.callAfter(_flash_start)
    except Exception as e:
        print(f"[alerte] flash impossible : {e!r}", flush=True)


def _flash_start():
    from AppKit import (NSApplication, NSBackingStoreBuffered, NSColor, NSScreen, NSWindow,
                        NSScreenSaverWindowLevel)
    from PyObjCTools import AppHelper
    for w in _flash_windows:
        w.orderOut_(None)
    _flash_windows.clear()
    for scr in NSScreen.screens():
        w = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(scr.frame(), 0, NSBackingStoreBuffered, False)
        w.setBackgroundColor_(NSColor.whiteColor())
        w.setOpaque_(False)
        w.setAlphaValue_(0.0)
        w.setLevel_(NSScreenSaverWindowLevel)
        w.setIgnoresMouseEvents_(True)
        w.setReleasedWhenClosed_(False)
        w.setCollectionBehavior_((1 << 0) | (1 << 4) | (1 << 8))  # tous les bureaux, fixe, par-dessus le plein écran
        w.orderFrontRegardless()
        _flash_windows.append(w)
    # deux éclairs francs, puis un fondu
    steps = [(0.00, 0.95), (0.12, 0.0), (0.24, 0.95), (0.36, 0.0), (0.48, 0.9)]
    steps += [(0.48 + 0.05 * k, 0.9 * (1 - k / 12)) for k in range(1, 13)]
    for t, alpha in steps:
        AppHelper.callLater(t, _flash_alpha, alpha)
    AppHelper.callLater(steps[-1][0] + 0.05, _flash_end)
    try:
        NSApplication.sharedApplication().requestUserAttention_(0)   # l'icône rebondit dans le Dock
    except Exception:
        pass


def _flash_alpha(alpha: float):
    for w in _flash_windows:
        w.setAlphaValue_(alpha)


def _flash_end():
    for w in _flash_windows:
        w.orderOut_(None)
    _flash_windows.clear()


# --------------------------------------------------------------------------- engine

class Engine:
    def __init__(self, cfg: Config, bus: EventBus, asr: Transcriber | None, voices: VoiceBook | None):
        self.cfg, self.bus, self.asr, self.voices = cfg, bus, asr, voices
        self.q: queue.Queue = queue.Queue()
        self.segmenters: list[ChannelSegmenter] = []
        log_dir = local_path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"com_{dt.datetime.now():%Y-%m-%d_%Hh%M}.txt"
        self.detector = load_detector(cfg)

    @property
    def mono(self) -> bool:
        return self.cfg.mode == "mono"

    def build(self, sr: int):
        if self.mono:
            sources = [Speaker(channel=self.cfg.channel, name="Com", gain_db=self.cfg.gain_db)]
        else:
            sources = self.cfg.speakers
        self.segmenters = [ChannelSegmenter(i, sp, self.cfg, sr, self._on_segment, self._on_state)
                           for i, sp in enumerate(sources)]

    def n_channels(self) -> int:
        return self.cfg.channel if self.mono else max(s.channel for s in self.cfg.speakers)

    def speakers_public(self) -> list[dict]:
        if self.mono:
            return self.voices.public() if self.voices else []
        return [{"name": s.name, "color": s.color, "alias": None, "named": True} for s in self.cfg.speakers]

    def _on_state(self, idx: int, talking: bool):
        self.bus.publish({"type": "activity" if self.mono else "talk", "speaker": idx, "talking": talking})

    def _on_segment(self, idx: int, audio: np.ndarray, t0: float):
        if self.asr is None:  # modèles encore en chargement
            return
        self.q.put((idx, audio, t0))

    def feed_block(self, block: np.ndarray, now: float | None = None):
        """block : (frames, n_channels) float32."""
        now = time.time() if now is None else now
        for seg in self.segmenters:
            ch = seg.speaker.channel - 1
            if ch < block.shape[1]:
                seg.feed(block[:, ch], now)

    def levels(self):
        return [round(s.level_db, 1) for s in self.segmenters]

    def _who(self, audio: np.ndarray, pieces: list[tuple[float, float, str]]) -> list[tuple[int, str]]:
        """Mode mono : attribue chaque morceau de phrase à une voix, puis regroupe les morceaux consécutifs."""
        out: list[tuple[int, str]] = []
        whole = None
        for start, end, text in pieces:
            a = audio[int(start * ASR_RATE): int(end * ASR_RATE)]
            if len(a) < ASR_RATE * 0.8:  # trop court pour une empreinte fiable : on prend la prise entière
                if whole is None:
                    whole = self.voices.identify(audio)
                spk, new = whole
            else:
                spk, new = self.voices.identify(a)
            if new:
                self.bus.publish({"type": "speakers", "speakers": self.voices.public()})
            if out and out[-1][0] == spk:
                out[-1] = (spk, out[-1][1] + " " + text)
            else:
                out.append((spk, text))
        return out

    busy = False

    def asr_loop(self):
        while True:
            item = self.q.get()
            self.busy = True
            try:
                self._process(*item)
            finally:
                self.busy = False

    def _process(self, idx: int, audio: np.ndarray, t0: float):
        t_start = time.time()
        try:
            pieces = [(s, e, clean_text(t)) for s, e, t in self.asr.transcribe(audio)]
            pieces = [p for k, p in enumerate(pieces)   # segments identiques à la suite = boucle de Whisper
                      if p[2] and (k == 0 or _fold(p[2])[0] != _fold(pieces[k - 1][2])[0])]
            if not pieces:
                return
            if self.mono:
                # les voix « ignorées » (ex. retour antenne) ne s'affichent pas
                lines = [(spk, t) for spk, t in self._who(audio, pieces) if not self.voices.is_ignored(spk)]
                if not lines:
                    return
            else:
                lines = [(idx, " ".join(p[2] for p in pieces))]
        except Exception as e:  # on ne plante jamais l'affichage pour une prise ratée
            print(f"[asr] erreur : {e}", file=sys.stderr)
            return
        spk_list = self.speakers_public()
        for spk, text in lines:
            sp = spk_list[spk]
            hits = self.detector.find(text)
            hit = bool(hits)
            if hit:
                flash_screen()
            ev = {
                "type": "line", "speaker": spk, "text": text,
                "time": dt.datetime.fromtimestamp(t0).strftime("%H:%M:%S"),
                "alert": hit, "hits": hits, "latency": round(time.time() - t0, 1),
                "asr_s": round(time.time() - t_start, 2),
            }
            self.bus.publish(ev)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(f"{ev['time']}  {sp['name']:>12} : {text}\n")
            print(f"{ev['time']}  {'🔔 ' if hit else ''}{sp['name']} : {text}", flush=True)


# --------------------------------------------------------------------------- sources audio

class LiveAudio:
    """Entrée audio en direct, choisie et changée depuis la fenêtre (carte son + entrée)."""

    def __init__(self, cfg: Config, eng: Engine):
        import sounddevice as sd
        self.sd, self.cfg, self.eng = sd, cfg, eng
        self.stream = None
        self.lock = threading.Lock()
        self.current: dict | None = None   # {device, name, channel, samplerate}
        self.error = ""

    # -- liste des cartes
    def devices(self, refresh: bool = False) -> list[dict]:
        if refresh:
            with self.lock:
                was = self.current
                self._stop()
                self.sd._terminate()
                self.sd._initialize()
            if was:
                self.start(was["name"], was["channel"])
        out = []
        for i, d in enumerate(self.sd.query_devices()):
            if d["max_input_channels"] > 0:
                out.append({"index": i, "name": d["name"], "inputs": int(d["max_input_channels"]),
                            "rate": int(d["default_samplerate"])})
        return out

    def diag(self) -> str:
        """Infos techniques affichées quand aucune carte n'est trouvée."""
        try:
            apis = ", ".join(h["name"] for h in self.sd.query_hostapis())
            alld = len(self.sd.query_devices())
            return (f"PortAudio {self.sd.get_portaudio_version()[1]} · pilotes : {apis or 'aucun'} · "
                    f"{alld} périphérique(s) audio vus au total · Python {sys.version.split()[0]} {sys.platform}")
        except Exception as e:
            return f"Erreur PortAudio : {e!r}"

    def _find(self, want) -> int | None:
        devs = self.devices()
        ok = {d["index"] for d in devs}
        if want in (None, "", "default"):
            try:
                i = int(self.sd.default.device[0])
                if i in ok:
                    return i
            except Exception:
                pass
            return devs[0]["index"] if devs else None
        if isinstance(want, int) or (isinstance(want, str) and want.isdigit()):
            return int(want) if int(want) in ok else None
        for d in devs:          # nom exact d'abord, puis morceau du nom
            if d["name"] == want:
                return d["index"]
        for d in devs:
            if str(want).lower() in d["name"].lower():
                return d["index"]
        return None

    # -- démarrage / changement
    def _stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.current = None

    def start(self, device, channel: int | None = None, save: bool = False) -> bool:
        try:
            return self._start(device, channel, save)
        except Exception as e:
            print(f"[audio] erreur à l'ouverture : {e!r}", file=sys.stderr, flush=True)
            self.error = f"Erreur à l'ouverture de la carte son : {e}"
            self._publish()
            return False

    def _start(self, device, channel, save) -> bool:
        idx = self._find(device)
        with self.lock:
            self._stop()
            if idx is None:
                self.error = (f"Carte son « {device} » introuvable : choisis-en une dans ⚙ Entrée." if device
                              else "Aucune carte son trouvée.")
                self._publish()
                return False
            info = self.sd.query_devices(idx)
            if self.eng.mono:
                ch = max(1, min(int(channel or self.cfg.channel), int(info["max_input_channels"])))
                self.cfg.channel = ch
            else:
                ch = None
            n_ch = self.eng.n_channels()
            if n_ch > info["max_input_channels"]:
                self.error = f"« {info['name']} » n'a que {info['max_input_channels']} entrée(s), il en faut {n_ch}."
                self._publish()
                return False
            last = None
            for sr in dict.fromkeys([self.cfg.samplerate, int(info["default_samplerate"]), 48000, 44100]):
                try:
                    self.eng.build(sr)
                    st = self.sd.InputStream(device=idx, channels=n_ch, samplerate=sr, dtype="float32",
                                             blocksize=int(sr * 0.03), callback=self._cb)
                    st.start()
                    self.stream = st
                    self.current = {"device": idx, "name": info["name"], "channel": ch, "samplerate": sr}
                    break
                except Exception as e:
                    last = e
            if self.stream is None:
                self.error = f"Impossible d'ouvrir « {info['name']} » : {last}"
                self._publish()
                return False
            self.error = ""
            print(f"[audio] {info['name']} — entrée {ch or n_ch} @ {self.current['samplerate']} Hz", flush=True)
            if save:
                save_settings(device=info["name"], channel=ch)
            self._publish()
            return True

    def _cb(self, indata, frames, t, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        self.eng.feed_block(indata.copy())

    def status(self) -> dict:
        return {"current": self.current, "error": self.error, "saved": "device" in load_settings()}

    def _publish(self):
        self.eng.bus.publish({"type": "audio", **self.status()})

    def run(self):
        saved = load_settings()
        dev, ch = saved.get("device", self.cfg.device), saved.get("channel") or self.cfg.channel
        self.start(dev, ch)
        while True:
            time.sleep(0.1)
            if self.stream is not None:
                self.eng.bus.publish({"type": "levels", "db": self.eng.levels()})


def run_file(cfg: Config, eng: Engine, path: str, speed: float):
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    print(f"[fichier] {path} — {data.shape[1]} canal(aux), {len(data) / sr:.1f}s @ {sr} Hz", flush=True)
    eng.build(sr)
    blk = int(sr * 0.03)
    t_next = t_file0 = time.time()
    for i in range(0, len(data), blk):
        eng.feed_block(data[i:i + blk], t_file0 + i / sr)
        if i % (blk * 4) == 0:
            eng.bus.publish({"type": "levels", "db": eng.levels()})
        if speed > 0:
            t_next += 0.03 / speed
            d = t_next - time.time()
            if d > 0:
                time.sleep(d)
    eng.feed_block(np.zeros((int(sr * 2), data.shape[1]), np.float32))  # vide les derniers segments
    time.sleep(0.5)
    while not eng.q.empty() or eng.busy:
        time.sleep(0.2)
    print("[fichier] terminé", flush=True)


# --------------------------------------------------------------------------- serveur web

def make_handler(cfg: Config, bus: EventBus, eng: Engine, live: "LiveAudio | None"):
    web = HERE / "web"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = (web / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/config":
                self._json({"mode": cfg.mode, "speakers": eng.speakers_public(), "live": live is not None,
                            "audio": live.status() if live else None, "status": bus.status,
                            "threshold_db": cfg.vad_threshold_db,
                            "keywords": cfg.keywords, "history": bus.history, "native": GUI_ACTIVE,
                            "alerte": {"actif": eng.detector.enabled, "prenoms": eng.detector.names,
                                       "mots": eng.detector.words}})
            elif self.path.startswith("/devices") and live:
                try:
                    devs = live.devices(refresh="refresh" in self.path)
                    self._json({"devices": devs, "diag": live.diag(), **live.status()})
                except Exception as e:
                    print(f"[audio] erreur liste des cartes : {e!r}", file=sys.stderr, flush=True)
                    self._json({"devices": [], "diag": f"Erreur : {e!r}", **live.status()})
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q = bus.subscribe()
                try:
                    while True:
                        try:
                            ev = q.get(timeout=15)
                            self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
                        except queue.Empty:
                            self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    bus.unsubscribe(q)
            else:
                self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/audio" and live:
                ok = live.start(int(req["device"]), req.get("channel"), save=True)
                self._json({"ok": ok, **live.status()})
            elif self.path == "/alerte":
                d = {"actif": bool(req.get("actif", True)),
                     "prenoms": [str(x)[:40] for x in req.get("prenoms", [])][:20],
                     "mots": [str(x)[:60] for x in req.get("mots", [])][:300]}
                save_settings(alerte=d)
                eng.detector = Detector(d["prenoms"], d["mots"], d["actif"])
                self._json({"ok": True})
            elif self.path == "/flash-test":
                flash_screen(force=True)
                bus.publish({"type": "flash"})
                self._json({"ok": True, "native": GUI_ACTIVE})
            elif self.path == "/open-transcripts":
                import subprocess
                local_path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(local_path(cfg.log_dir))])
                self._json({"ok": True})
            elif self.path == "/quit":
                self._json({"ok": True})
                threading.Timer(0.3, lambda: os._exit(0)).start()
            elif self.path == "/voice" and eng.voices:
                eng.voices.act(int(req["speaker"]), str(req.get("action", "")), str(req.get("name", "")),
                               req.get("into"))
                bus.publish({"type": "speakers", "speakers": eng.voices.public()})
                self._json({"ok": True})
            elif self.path == "/vad":
                cfg.vad_threshold_db = max(-80.0, min(0.0, float(req["threshold_db"])))
                save_settings(threshold_db=cfg.vad_threshold_db)
                bus.publish({"type": "vad", "threshold_db": cfg.vad_threshold_db})
                self._json({"ok": True, "threshold_db": cfg.vad_threshold_db})
            elif self.path == "/rename" and eng.voices:
                eng.voices.act(int(req["speaker"]), "rename", str(req["name"]))
                spk = eng.voices.public()
                bus.publish({"type": "speakers", "speakers": spk})
                self._json({"ok": True})
            else:
                self.send_error(404)

    return H


# --------------------------------------------------------------------------- app Mac, fenêtre, mises à jour

APP_BUNDLE = Path.home() / "Applications" / "Intercom Live.app"
RESTART_CODE = 42   # lancer.command relance le logiciel quand il sort avec ce code


def open_window(url: str):
    """Ouvre l'interface : fenêtre d'app Chrome si dispo (sans onglets ni barre d'adresse), sinon le navigateur."""
    if sys.platform == "darwin":
        import subprocess
        for app in ("Google Chrome", "Microsoft Edge", "Brave Browser"):
            if Path(f"/Applications/{app}.app").exists():
                subprocess.Popen(["open", "-na", app, "--args", f"--app={url}", "--window-size=1100,750"])
                return
    webbrowser.open(url)


APP_MODE = False   # lancé depuis l'icône (pas de Terminal) : les redémarrages se font sans lancer.command


def native_window_available() -> bool:
    if sys.platform != "darwin" or os.environ.get("INTERCOM_LIVE_NO_GUI"):
        return False
    try:
        import webview  # noqa: F401 (pywebview : fenêtre Mac native, WebKit comme Safari)
        return True
    except Exception as e:
        print(f"[fenêtre] fenêtre native indisponible ({e!r}), j'utilise le navigateur.", flush=True)
        return False


def run_native_window(url: str):
    import webview
    try:  # nom « Intercom Live » dans la barre de menus et le Dock (sinon « Python »)
        from Foundation import NSBundle
        info = NSBundle.mainBundle().infoDictionary()
        info["CFBundleName"] = "Intercom Live"
        info["CFBundleDisplayName"] = "Intercom Live"
    except Exception:
        pass

    def set_icon():
        try:
            from AppKit import NSApplication, NSImage
            icns = APP_BUNDLE / "Contents" / "Resources" / "icon.icns"
            if icns.exists():
                NSApplication.sharedApplication().setApplicationIconImage_(
                    NSImage.alloc().initWithContentsOfFile_(str(icns)))
        except Exception:
            pass

    global GUI_ACTIVE
    GUI_ACTIVE = True
    webview.create_window("Intercom Live", url, width=1100, height=750, min_size=(640, 420),
                          background_color="#0e1013", text_select=True)
    webview.start(set_icon)


def ensure_requirements():
    """Installe les modules ajoutés par une mise à jour (utilisé quand l'app se relance elle-même)."""
    import hashlib
    import subprocess
    req = HERE / "requirements.txt"
    stamp = HERE / ".venv" / ".req_hash"
    h = hashlib.sha1(req.read_bytes()).hexdigest()
    if stamp.exists() and stamp.read_text().strip() == h:
        return
    print("[maj] installation des nouveaux modules…", flush=True)
    py = sys.executable
    if subprocess.run([py, "-m", "pip", "--version"], capture_output=True).returncode == 0:
        cmd = [py, "-m", "pip", "install", "-r", str(req)]
    else:
        uv = shutil_which("uv") or str(Path.home() / ".local" / "bin" / "uv")
        cmd = [uv, "pip", "install", "--python", py, "-r", str(req)]
    if subprocess.run(cmd).returncode == 0:
        stamp.write_text(h)


def shutil_which(name):
    import shutil
    return shutil.which(name)


def restart_self():
    if APP_MODE:   # même processus (l'app garde son icône et son accès au micro)
        try:
            ensure_requirements()
        except Exception as e:
            print(f"[maj] modules : {e!r}", flush=True)
        os.execv(sys.executable, [sys.executable, "-u", str(HERE / "intercom_live.py")] + sys.argv[1:])
    os._exit(RESTART_CODE)   # lancé depuis le Terminal : lancer.command relance


def already_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/config", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


def _png(size: int) -> bytes:
    """Icône dessinée à la main (pas de dépendance) : carré arrondi sombre, bulle et voyant rouge."""
    import struct
    import zlib
    y, x = np.mgrid[0:size, 0:size] / size
    img = np.zeros((size, size, 4), np.float32)
    r = 0.22
    cx, cy = np.clip(x, r, 1 - r), np.clip(y, r, 1 - r)
    inside = ((x - cx) ** 2 + (y - cy) ** 2) <= r * r
    inside &= (x > 0.04) & (x < 0.96) & (y > 0.04) & (y < 0.96)
    img[inside] = (0.09, 0.10, 0.13, 1)
    for i, (y0, w) in enumerate(((0.36, 0.56), (0.50, 0.44), (0.64, 0.30))):  # lignes de texte
        m = (np.abs(y - y0) < 0.035) & (x > 0.2) & (x < 0.2 + w) & inside
        img[m] = ((1.0, 0.44, 0.26, 1), (0.16, 0.71, 0.96, 1), (0.40, 0.73, 0.42, 1))[i]
    dot = ((x - 0.76) ** 2 + (y - 0.24) ** 2) < 0.07 ** 2
    img[dot] = (0.90, 0.22, 0.21, 1)
    raw = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    rows = b"".join(b"\x00" + raw[i].tobytes() for i in range(size))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def install_app_bundle() -> Path:
    """Crée ~/Applications/Intercom Live.app, qui lance ce dossier sans Terminal."""
    import plistlib
    import subprocess
    contents = APP_BUNDLE / "Contents"
    (contents / "MacOS").mkdir(parents=True, exist_ok=True)
    (contents / "Resources").mkdir(exist_ok=True)
    plistlib.dump({
        "CFBundleName": "Intercom Live", "CFBundleDisplayName": "Intercom Live",
        "CFBundleIdentifier": "fr.intercomlive.app", "CFBundleExecutable": "IntercomLive",
        "CFBundleIconFile": "icon", "CFBundlePackageType": "APPL", "CFBundleShortVersionString": "1.0",
        "LSMinimumSystemVersion": "12.0",
        "NSMicrophoneUsageDescription": "Intercom Live écoute la com de régie pour l'écrire en direct.",
    }, (contents / "Info.plist").open("wb"))
    exe = contents / "MacOS" / "IntercomLive"
    exe.write_text(f'#!/bin/bash\nexec /bin/bash "{HERE_FOR_BUNDLE / "lancer.command"}" --app\n', encoding="utf-8")
    exe.chmod(0o755)
    try:  # icône (best effort)
        iconset = DATA_DIR / "icon.iconset"
        iconset.mkdir(parents=True, exist_ok=True)
        for sz in (16, 32, 128, 256, 512):
            (iconset / f"icon_{sz}x{sz}.png").write_bytes(_png(sz))
            (iconset / f"icon_{sz}x{sz}@2x.png").write_bytes(_png(sz * 2))
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(contents / "Resources" / "icon.icns")],
                       check=True, capture_output=True)
    except Exception as e:
        print(f"[app] icône non créée : {e!r}", flush=True)
    subprocess.run(["touch", str(APP_BUNDLE)])
    print(f"[app] créée : {APP_BUNDLE}", flush=True)
    return APP_BUNDLE


UPDATE_FILES = ["intercom_live.py", "requirements.txt", "lancer.command", "config.toml", "web/index.html"]


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _gh_raw(repo: str, path: str, ref: str = "main") -> bytes:
    """Lit un fichier du dépôt GitHub public (API : pas de cache, donc la version est toujours la dernière)."""
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}",
                                 headers={"Accept": "application/vnd.github.raw", "User-Agent": "intercom-live"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read()


def auto_update(repo: str, every_s: int = 120):
    """Télécharge les nouvelles versions publiées sur GitHub. Le fichier remplacé déclenche ensuite
    watch_for_updates (rechargement de la page ou redémarrage)."""
    import py_compile
    import tempfile
    time.sleep(10)
    while True:
        try:
            man = json.loads(_gh_raw(repo, "manifest.json"))
            changed = [f for f, h in man.get("files", {}).items() if f in UPDATE_FILES and _sha(HERE / f) != h]
            if changed:
                print(f"[maj] nouvelle version {man.get('version', '?')} : {', '.join(changed)}", flush=True)
                staged = {}
                for f in changed:   # tout télécharger et vérifier avant de remplacer quoi que ce soit
                    data = _gh_raw(repo, f, man.get("commit", "main"))
                    import hashlib
                    if hashlib.sha256(data).hexdigest() != man["files"][f]:
                        raise ValueError(f"empreinte de {f} incorrecte")
                    staged[f] = data
                if "intercom_live.py" in staged:
                    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as t:
                        t.write(staged["intercom_live.py"])
                    py_compile.compile(t.name, doraise=True)
                    os.unlink(t.name)
                # le script de lancement d'abord, le code Python en dernier (c'est lui qui déclenche le redémarrage)
                for f in sorted(staged, key=lambda f: f == "intercom_live.py"):
                    dst = HERE / f
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dst.with_name(dst.name + ".maj")
                    tmp.write_bytes(staged[f])
                    if f.endswith(".command"):
                        tmp.chmod(0o755)
                    os.replace(tmp, dst)   # remplacement atomique (ne casse pas le script en cours)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                print(f"[maj] vérification impossible : {e}", flush=True)
        except Exception as e:
            print(f"[maj] vérification impossible : {e!r}", flush=True)
        time.sleep(every_s)


APP_HOME = DATA_DIR / "app"   # emplacement de l'app que macOS laisse lire quand on la lance depuis le Dock


def relocate_out_of_documents(bus: "EventBus | None" = None) -> bool:
    """La v1.0/1.1 a pu s'installer dans ~/Documents : macOS interdit alors à l'app du Dock de lire ses
    fichiers. Lancé depuis le Terminal (qui, lui, y a accès), on recopie l'app au bon endroit et on
    refait l'icône. Renvoie True si l'app a été déplacée."""
    import shutil
    if sys.platform != "darwin" or HERE == APP_HOME:
        return False
    try:
        HERE.relative_to(Path.home() / "Documents")
    except ValueError:
        return False
    print(f"[app] déplacement de l'app vers {APP_HOME} (macOS bloque Documents pour l'app du Dock)…", flush=True)
    APP_HOME.mkdir(parents=True, exist_ok=True)
    for item in HERE.iterdir():
        if item.name in ("dernier_lancement.log",) or item.suffix == ".maj":
            continue
        if item.name == ".venv" and (APP_HOME / ".venv" / "bin" / "python").exists():
            continue   # déjà recopié lors d'un lancement précédent
        dst = APP_HOME / item.name
        if item.is_dir():
            shutil.copytree(item, dst, symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)
    (APP_HOME / "lancer.command").chmod(0o755)
    global HERE_FOR_BUNDLE
    HERE_FOR_BUNDLE = APP_HOME
    install_app_bundle()
    print("[app] ✅ app réparée : la prochaine fois, lance « Intercom Live » depuis le Dock.", flush=True)
    if bus:
        bus.set_status("✅ App réparée : ferme le Terminal quand tu veux, et la prochaine fois lance « Intercom Live » "
                       "depuis le Dock ou le Launchpad.")
    return True


HERE_FOR_BUNDLE = HERE


def watch_for_updates(bus: "EventBus"):
    """Quand le code est mis à jour (par Claude), recharge la page ou relance le logiciel tout seul."""
    import py_compile
    code_files = [HERE / "intercom_live.py", HERE / "requirements.txt", HERE / "config.toml"]
    web_files = list((HERE / "web").glob("*"))
    stamp = lambda fs: tuple(f.stat().st_mtime if f.exists() else 0 for f in fs)
    code0, web0 = stamp(code_files), stamp(web_files)
    while True:
        time.sleep(2)
        try:
            web1 = stamp(web_files)
            if web1 != web0:
                time.sleep(1.5)                     # on attend la fin de l'écriture
                web0 = stamp(web_files)
                print("[maj] interface mise à jour", flush=True)
                bus.publish({"type": "reload"})
            code1 = stamp(code_files)
            if code1 != code0:
                time.sleep(2)
                if stamp(code_files) != code1:      # encore en cours d'écriture
                    continue
                try:
                    py_compile.compile(str(HERE / "intercom_live.py"), doraise=True)
                    tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
                except Exception as e:      # mise à jour incomplète ou cassée : on garde la version en cours
                    print(f"[maj] nouvelle version illisible, on attend : {e}", flush=True)
                    code0 = code1
                    continue
                print("[maj] nouvelle version, redémarrage…", flush=True)
                bus.publish({"type": "restarting"})
                time.sleep(0.5)
                restart_self()
        except Exception as e:
            print(f"[maj] {e!r}", flush=True)


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Transcription live de la com intercom")
    ap.add_argument("--config", default=str(HERE / "config.toml"))
    ap.add_argument("--list-devices", action="store_true", help="liste les interfaces audio")
    ap.add_argument("--file", help="rejoue un fichier audio au lieu du live")
    ap.add_argument("--speed", type=float, default=1.0, help="vitesse de lecture du fichier (0 = max)")
    ap.add_argument("--fake-asr", action="store_true", help="sans Whisper (test de la chaîne)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--app", action="store_true", help=argparse.SUPPRESS)          # lancé par l'app Mac
    ap.add_argument("--install-app", action="store_true", help="crée l'app Mac dans ~/Applications")
    a = ap.parse_args()

    global APP_MODE
    APP_MODE = a.app
    if a.install_app:
        _migrate_old_data()
        if not relocate_out_of_documents():
            install_app_bundle()
        return

    if a.list_devices:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"{i:3d}  {d['name']}  ({d['max_input_channels']} entrées, {int(d['default_samplerate'])} Hz)")
        return

    port = Config.load(Path(a.config)).port
    if already_running(port):   # déjà ouvert (ex. 2e clic sur l'icône) : on montre juste la fenêtre
        print("Intercom Live tourne déjà : j'ouvre la fenêtre.", flush=True)
        if sys.platform == "darwin" and APP_BUNDLE.exists():
            import subprocess
            subprocess.run(["open", str(APP_BUNDLE)])
        elif not a.no_browser:
            open_window(f"http://127.0.0.1:{port}")
        return

    _migrate_old_data()
    print(f"[données] voix, réglages et transcriptions dans : {DATA_DIR}", flush=True)
    cfg = Config.load(Path(a.config))
    if "threshold_db" in load_settings():
        cfg.vad_threshold_db = float(load_settings()["threshold_db"])
    if cfg.mode == "multi" and not cfg.speakers:
        sys.exit("Mode multi : aucun poste dans config.toml ([[speakers]]).")
    bus = EventBus()
    eng = Engine(cfg, bus, None, None)
    live = None if a.file else LiveAudio(cfg, eng)

    try:
        srv = ThreadingHTTPServer(("127.0.0.1", cfg.port), make_handler(cfg, bus, eng, live))
    except OSError:
        sys.exit(f"Le port {cfg.port} est déjà pris : Intercom Live tourne sans doute déjà (ou change [ui] port).")
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{cfg.port}"
    print(f"[ui] {url}   (journal : {eng.log_path})", flush=True)
    use_gui = cfg.open_browser and native_window_available()
    threading.Thread(target=watch_for_updates, args=(bus,), daemon=True).start()
    if cfg.update_repo and not a.file:
        threading.Thread(target=auto_update, args=(cfg.update_repo,), daemon=True).start()

    def boot():
        if live:
            try:
                devs = live.devices()
                print(f"[audio] {len(devs)} carte(s) son avec entrée : " + (", ".join(d["name"] for d in devs) or "AUCUNE"), flush=True)
                if not devs:
                    print(f"[audio] diagnostic : {live.diag()}", flush=True)
            except Exception as e:
                print(f"[audio] impossible de lister les cartes son : {e!r}", flush=True)
            threading.Thread(target=live.run, daemon=True).start()

        # Chargement des modèles une fois la fenêtre ouverte (le 1er lancement télécharge ~1,5 Go).
        if cfg.mode == "mono":
            bus.set_status("Chargement de la reconnaissance des voix…")
        try:
            if cfg.mode == "mono":
                eng.voices = VoiceBook(cfg)
            bus.set_status("Chargement de Whisper (le premier lancement télécharge le modèle, quelques minutes)…")
            asr = Transcriber(cfg, fake=a.fake_asr)
            asr.warmup()
            eng.asr = asr
            bus.set_status("")
            threading.Thread(target=eng.asr_loop, daemon=True).start()
        except Exception as e:
            # On garde la fenêtre et l'audio en vie pour pouvoir lire l'erreur et régler l'entrée.
            import traceback
            traceback.print_exc()
            eng.asr = None
            bus.set_status(f"⚠ Erreur au chargement des modèles : {e!r}. Le détail est dans la fenêtre Terminal "
                           f"et dans dernier_lancement.log (dossier de l'app).")

        if not a.app and not a.file:   # ancienne installation dans Documents → on répare l'app du Dock
            threading.Thread(target=relocate_out_of_documents, args=(bus,), daemon=True).start()

    boot_thread = threading.Thread(target=boot, daemon=True)
    if use_gui:
        # Vraie fenêtre Mac : l'interface Cocoa doit tourner sur le fil principal, le reste en arrière-plan.
        boot_thread.start()
        if a.file:
            threading.Thread(target=run_file, args=(cfg, eng, a.file, a.speed), daemon=True).start()
        run_native_window(url)
        print("Fenêtre fermée : arrêt.", flush=True)
        os._exit(0)
    if cfg.open_browser and not a.no_browser:
        open_window(url)
    boot()
    try:
        if a.file:
            run_file(cfg, eng, a.file, a.speed)
            print("La fenêtre reste ouverte (renommer les voix, relire). Ctrl+C pour quitter.", flush=True)
            while True:
                time.sleep(1)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nArrêt.")


if __name__ == "__main__":
    main()
