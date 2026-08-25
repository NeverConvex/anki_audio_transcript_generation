# Standard libraries
import glob, json, os, pathlib

# Non-standard libraries
import fire
from transformers import pipeline

def transcribeSingleAudioFile(transcriber, audio_file_path, language="japanese"):
    result = transcriber(
        audio_file_path,
        generate_kwargs={"language": language},
        return_timestamps=True
    )
    return result

def transcribeAudioFiles(audio_fpaths_input_glob_expr, output_jsonl_fpath, output_mode="a", test_len=None):
    print(f"Received audio input fpaths glob expr: {audio_fpaths_input_glob_expr}")
    print("Initializing transcriber as Hugging Faces pipeline...")
    transcriber = pipeline(
        task="automatic-speech-recognition", model="openai/whisper-large-v3"
    )

    # NOTE our initial application involves running this on a highly unstable old laptop, so it is critical to checkpoint/restart from checkpoints
    data = []
    if os.path.isfile(output_jsonl_fpath):
        with open(output_jsonl_fpath, encoding="utf-8") as rf:
            data = [json.loads(line) for line in rf.readlines()]

    audio_fpaths = glob.glob(audio_fpaths_input_glob_expr)
    audio_fpaths = audio_fpaths[:test_len] if test_len is not None else audio_fpaths

    with open(output_jsonl_fpath, encoding='utf8', mode=output_mode) as wf:
        for index, audio_fp in enumerate(audio_fpaths):
            if index >= len(data) - 1:
                print(f"Beginning transcription for audio file # {index+1} of {len(audio_fpaths)}: {audio_fp}")
                result = transcribeSingleAudioFile(transcriber, audio_fp, language="japanese")
                print(f"\tWhisper returned: {result}")
                result["source_file"] = pathlib.Path(audio_fp).stem + pathlib.Path(audio_fp).stem
                wf.write(json.dumps(result, ensure_ascii=False) + '\n')
                wf.flush()
                os.fsync(wf.fileno()) # flush + fsync to try to be especially certain we write to checkpointing file before moving on
                print(f"\tDumped Whisper transcription to {output_jsonl_fpath}: {result}")

def generateLLMTranslations():
    raise NotImplementedError()

if __name__ == "__main__":
    # NOTE Example run command:
    # python transcribe_audio_w_translations.py transcribeAudioFiles --audio_fpaths_input_glob_expr="audio_files/Mixed_Exam_Audio_25_08_2026/*.mp3" --output_jsonl_fpath="mixed_exam_audio_whisper_test.jsonl" --test_len=3
    fire.Fire()
