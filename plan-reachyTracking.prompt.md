Plan: Brainstorm And Prioritize Additions

TL;DR: Propose a curated, high‑impact set of ideas tailored to this codebase across features (target selection UI, patrol mode, speech reactions, persistence), performance (GPU/DirectML, process‑isolated DNN, adaptive rates), robustness (thread health, locking, camera reconnect, input validation), UX (status clarity, live logs, manual control polish), docs (Windows steps, tuning), testing/CI (mock shim, tracker/tests), and DevX (version pins, env config, logging, WebSockets). Validate constraints, rank into Now/Next/Later, define quick spikes, and align on a short roadmap.

Goals
- Map constraints (GPU/OS/network), align scope from requirements and camera/robot assumptions.
- Draft concrete, scoped ideas with code touchpoints across backend (FastAPI + workers), detection, robot shim, and dashboard.
- Prioritize into Now/Next/Later by impact vs. effort; define success metrics.
- Produce a short, actionable roadmap and test plan for top picks.

Steps
1) Confirm constraints and goals: GPU availability (CUDA/DirectML), OS targets (Windows/macOS), network exposure (localhost vs. LAN), robot daemon contract.
2) Draft and group concrete ideas with file/symbol touchpoints, focusing on:
   - Backend: routes in main server (status, tuning, control), worker orchestration, shared state.
   - Detection: ONNX/YOLO pipeline, post‑proc, adaptive cadence.
   - Robot control: shim calls, gating, safety clamps, command queueing.
   - UI: dashboard status, controls, tuning panel, candidate list and interactions.
3) Prioritize ideas (impact/effort) → Now (quick wins, 1–3 days), Next (1–2 weeks), Later (2–4+ weeks).
4) Define success criteria and 0.5–1 day spikes for top picks (demoable with measurable outcomes: FPS, latency, stability).
5) Produce a lightweight roadmap and minimal test plan (unit + smoke) for selected items; stage tasks for implementation.

Further Considerations
- Acceleration strategy: CUDA vs. DirectML vs. CPU‑only; choose based on target hardware.
- Exposure model: local‑only vs. add simple auth if remote access required.
- Persistence: JSON config and/or .env to persist tuning and UI prefs.

Deliverables
- Curated idea list with file/symbol touchpoints and rationale.
- Prioritized Now/Next/Later backlog with rough effort and risk notes.
- Success metrics for top items (e.g., DNN FPS, end‑to‑end latency, recovery time).
- Mini test plan for top items (unit boundaries, mocks, smoke checks).


Initial Idea Categories (for ranking)
- Features: Target selection UI (click‑to‑track), patrol/auto‑scan mode, event‑based speech reactions, tuning persistence.
- Voice & AI (New): 
    - **Audio Pipeline**: Use `MediaManager` (from `reachy_mini` SDK) for microphone access (`start_recording`, `get_audio_sample`). [Ref: test_audio.py](https://github.com/pollen-robotics/reachy_mini/blob/develop/tests/test_audio.py)
    - **STT**: Local OpenAI Whisper (e.g., `faster-whisper`) for speech-to-text.
    - **LLM**: Local Qwen3 1.7B (lightweight, "Thinking Mode" capable) for reasoning and response generation. [Ref: Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
    - **TTS**: Local Piper TTS (fast, low-latency, Python bindings) for voice response. Voice: [Amy (Medium)](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/amy/medium).
    - **Flow**: Wake word/VAD -> Record -> Whisper -> Qwen3 -> Piper TTS -> Play Audio.
- Performance: GPU/DirectML or onnxruntime EPs, process‑isolated DNN, adaptive rate/resolution.
- Robustness: Thread health/restarts, stronger locking around shared state, camera reconnect/backoff, input validation on tuning/control.
- UX: Status enrichment (current target, age, FPS), live logs panel, manual control nudge/sensitivity, fix candidate highlighting.
- Docs: Windows setup, robot daemon prerequisites/ports, tuning guide, troubleshooting.
- Testing/CI: Mocked robot shim, tracker/detection post‑proc tests, control gating tests, lint/type checks.
- DevX: Version pins, env config, structured logging, WebSocket telemetry for status/logs.

Open Questions
- Hardware: Is a CUDA/DML‑capable GPU available on target machines?
- Network: Should the dashboard remain localhost only, or support LAN with auth?
- Persistence: Preferred mechanism (JSON, .env, or API‑backed storage)?
- Control: Confirm units and safety limits for head/body/antennas in the robot daemon.
- Target priorities: Person vs. face precedence when both are present?
- Voice/AI: Preferred runtime for Qwen3 (Transformers vs. Llama.cpp)?

Next Actions
- Validate constraints with stakeholders.
- Rank the categories and pick 2–3 Now items for immediate spikes.
- Outline acceptance criteria and quick test plans for chosen items.

Next Steps (Actionable Roadmap)
Now (0–2 days)
- Voice wake/VAD: Add robust VAD with webrtcvad; optional wake keyword (“Hey Reachy”). Success: <300 ms trigger latency; <1% false positives in quiet room.
- STT tuning: Optimize Whisper chunking and silence end-detection; cache model. Success: End-to-end speech → text in <2.5 s for 5 s utterance.
- TTS stability: Confirm Piper playback path via `play_sound_from_file()` pauses camera reliably; add retry on device busy. Success: 100% playback without OpenCV errors across 10 trials.
- UI polish: Voice panel status indicators (enabled/listening/loading), error toasts. Success: Clear feedback on model load and mic state.
- Config: Introduce `.env` overrides for `REACHY_HOST`, `VOICE_ENABLED`, `MODEL_DIR`. Success: App reads overrides at startup.

Next (1–2 weeks)
- Command grammar: Map common voice intents to robot actions (look left/right, recenter, toggle wiggle, set motor mode). Files: `main.py` (intent router), `robot_controller.py`. Success: 10+ commands with confirmation and safety clamps.
- GPU acceleration: Enable CUDA/DirectML for Whisper/Qwen3 where available or switch STT to `whisper-tiny.en` if CPU-only. Success: 40–60% latency reduction on supported hardware.
- Process isolation: Run STT/LLM/TTS in a separate process to avoid blocking tracking; use IPC queues. Files: `voice_assistant.py`, new `voice_worker.py`. Success: No missed video frames during long generations.
- Persistence: Save tuning + voice prefs to JSON; restore on startup. Files: `main.py`, `config.json`. Success: Settings survive restarts.
- WebSocket telemetry: Stream status/candidates/voice events via WS; reduce polling. Files: `main.py`, `static/index.html`. Success: <200 ms UI update latency.

Later (2–4+ weeks)
- Wake word model: Integrate Porcupine/KWS for “Hey Reachy”. Success: <1% FAR, <5% FRR in household noise.
- Multilingual: Add Whisper multilingual; Piper voice packs; language auto-detect. Success: English + one additional language end-to-end.
- Skills & memory: Add long-term memory (SQLite) for facts and user preferences; skill plugins (reminders, jokes, facts). Success: 3+ skills with persistence.
- Test suite & CI: Unit tests for VAD/STT/intent mapping; mocked SDK; GitHub Actions pipeline. Success: Green CI on PRs; 80% coverage on voice core.

Acceptance Criteria (Top items)
- Voice loop stability: No crashes across 30 minutes continuous use; recover after device errors without restart.
- Latency: STT+LLM+TTS end-to-end <3.5 s on CPU for typical utterance (~5 s).
- Robot safety: Intent routes clamp angles/speeds; manual mode required for hazardous moves.
- UX clarity: UI reflects voice/mic/model state; logs show concise INFO/WARN/ERROR for voice pipeline.

Dependencies & Risks
- Model size (Qwen3 ~3 GB): consider smaller model fallback if RAM < 12 GB.
- Piper availability on Windows: require PATH setup; provide Python binding fallback.
- Microphone access conflicts: ensure exclusive device handling; backoff + retry.

Known Issues
- **Long Audio Playback Truncation**: Audio clips >30 seconds may stop playing before completion despite correct duration calculations. The SDK's `MediaManager.push_audio_sample()` successfully queues all samples (~694K samples for 31.5s audio), but playback terminates early. Root cause investigation needed:
  - SDK buffer underrun: Media pipeline may not buffer properly for extended clips
  - Context manager lifecycle: `with SDKReachyMini()` block might release resources prematurely
  - Chunk pacing: Current 1024-sample chunks pushed as fast as possible may overwhelm buffer
  - Motion thread timing: Speaking animation thread may timeout before playback completes
  - Potential solutions: Monitor SDK buffer status, add inter-chunk delays, use SDK's internal playback completion signals if available, or chunk long responses into multiple <20s segments. Current workaround: Limit response length in LLM system prompt.
