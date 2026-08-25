# Standard libraries
import glob, json, os

# Non-standard libraries
import fire
from transformers import pipeline

def transcribeSingleAudioFile(transcriber, audio_file_path, language="japanese")
    result = transcriber(
        audio_file_path,
        generate_kwargs={"language": language},
        return_timestamps=True
    )
    return result

def transcribeAudioFiles(audio_fpaths_input_glob_expr, output_jsonl_fpath, output_mode="a"):
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

    with open(output_jsonl_fpath, encoding='utf8', mode=putput_mode) as wf:
        for index, audio_fp in enumerate(glob.glob(audio_fpaths_input_glob_expr)):
            if index >= len(data) - 1:
                print(f"Beginning transcription of: {audio_fp}")
                result = transcribeSingleAudioFile(transcriber, audio_fp, language="japanese")
                wf.write(json.dumps(result, ensure_ascii=False) + '\n') 
                print(f"Dumped Whisper transcription to {output_jsonl_fpath}: {result}")

def generateLLMTranslations():
    raise NotImplementedError()

if __name__ == "__main__":
    fire.Fire()
