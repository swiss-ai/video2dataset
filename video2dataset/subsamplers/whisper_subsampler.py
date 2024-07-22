"""
Whisper subsampler - transcribes audio using the Whisper model from OAI using WhisperX API

code: https://github.com/m-bain/whisperX
"""

import os
import time
import tempfile

try:
    import whisperx
    import torch
except:  # pylint: disable=broad-except,bare-except
    pass

from .subsampler import Subsampler


class WhisperSubsampler(Subsampler):
    """
    Transcribes audio samples using the OAI Whisper Model via WhisperX API

    Params:
        model_name: https://github.com/guillaumekln/faster-whisper/blob/20d4e9418b5efb69ec5aa4819a39e3fb0e772a2a/faster_whisper/transcribe.py#LL90C1-L90C1
        batch_size: batch size used during inference (try to maximize this for perf)
        compute_type: accuracy/mem tradeoff (float16, float32, int8)
    """

    def __init__(
        self,
        model_name="large-v2",
        batch_size=16,
        compute_type="float16",
        download_root=None,
        is_slurm_task=False,
        force_cpu=False,
        language=None,
        align=False,
    ):
        print("INIT WHISPER")
        self.language = language
        if is_slurm_task:
            global_rank = os.environ["GLOBAL_RANK"]
            if global_rank != 0:
                time.sleep(20)  # let master worker download model

            self.device, device_index = "cuda", int(os.environ["LOCAL_RANK"])
            if force_cpu:
                self.device = "cpu"
                device_index = 0
            options = {
                "max_new_tokens": None,
                "clip_timestamps": None,
                "hallucination_silence_threshold": None,
                "hotwords": None,
                # "word_timestamps": True,
                # "without_timestamps": False
            }
            while True:
                try:
                    self.model = whisperx.load_model(
                        model_name,
                        device=self.device,
                        device_index=device_index,
                        compute_type=compute_type,
                        download_root=download_root,
                        language=self.language,
                        asr_options=options,
                    )
                    print("model_loaded", os.environ["GLOBAL_RANK"], flush=True)
                    if not self.language:
                        raise ValueError("Alignment needs a language code!")
                    if align:
                        self.align_model, self.align_metadata = (
                            whisperx.load_align_model(
                                language_code=self.language, device=self.device
                            )
                        )
                        print(
                            "align_model_loaded", os.environ["GLOBAL_RANK"], flush=True
                        )
                        print(self.align_model, self.align_metadata)
                    else:
                        self.align_model = None

                    break
                except Exception as e:  # pylint: disable=(broad-except)
                    print(str(e), flush=True)
                    print(
                        "loading failed, retrying...",
                        os.environ["GLOBAL_RANK"],
                        flush=True,
                    )
                    continue
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.model = whisperx.load_model(
                model_name,
                device=self.device,
                compute_type=compute_type,
                download_root=download_root,
                language=self.language,
                asr_options=options,
            )
            if align:
                if not self.language:
                    raise ValueError("Alignment needs a language code!")
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=self.language, device=self.device
                )

        self.batch_size = batch_size

    def __call__(self, streams, metadata=None):
        audio_bytes = streams.get("audio")

        for i, aud_bytes in enumerate(audio_bytes):
            # TODO: .m4a not always
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3") as tmpfile:
                    tmpfile.write(aud_bytes)
                    tmpfile.flush()  # ensure all data is written
                    audio = whisperx.load_audio(tmpfile.name)
                    print("run whisper!")

                    result = self.model.transcribe(
                        audio, batch_size=self.batch_size, language=self.language
                    )
                    metadata[i]["whisper_transcript"] = result
                    print(result)

                    if self.align_model:
                        align_result = whisperx.align(
                            result["segments"],
                            self.align_model,
                            self.align_metadata,
                            audio,
                            self.device,
                            return_char_alignments=False,
                        )
                        print(align_result)
                        metadata[i]["whisper_alignment"] = align_result
            except Exception as err:  # pylint: disable=broad-except
                return [], metadata, str(err)

        return streams, metadata, None
