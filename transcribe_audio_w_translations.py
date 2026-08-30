# Standard libraries
import glob, json, os, pathlib

# Non-standard libraries
import fire
from transformers import pipeline
from openai import OpenAI

def transcribeSingleAudioFile(transcriber, audio_file_path, language="japanese"):
    result = transcriber(
        audio_file_path,
        generate_kwargs={"language": language},
        return_timestamps=True
    )
    return result

def transcribeAudioFiles(audio_fpaths_input_glob_expr, output_jsonl_fpath, test_len=None, get_llm_translations=False):
    print(f"Received audio input fpaths glob expr: {audio_fpaths_input_glob_expr}")
    print("Initializing transcriber as Hugging Faces pipeline...")
    transcriber = pipeline(
        task="automatic-speech-recognition", model="openai/whisper-large-v3"
    )

    # NOTE our initial application involves running this on a highly unstable old laptop, so it is critical to checkpoint/restart from checkpoints
    data = []
    if os.path.isfile(output_jsonl_fpath):
        with open(output_jsonl_fpath, encoding="utf-8", mode='r') as rf:
            data = [json.loads(line) for line in rf.readlines()]

    audio_fpaths = glob.glob(audio_fpaths_input_glob_expr)
    audio_fpaths = audio_fpaths[:test_len] if test_len is not None else audio_fpaths

    with open(output_jsonl_fpath, encoding='utf8', mode='a') as wf:
        for index, audio_fp in enumerate(audio_fpaths):
            if index >= len(data):
                print(f"Beginning transcription for audio file # {index+1} of {len(audio_fpaths)}: {audio_fp}")
                result = transcribeSingleAudioFile(transcriber, audio_fp, language="japanese")
                print(f"\tWhisper returned: {result}")
                result["source_file"] = pathlib.Path(audio_fp).stem + pathlib.Path(audio_fp).stem
                wf.write(json.dumps(result, ensure_ascii=False) + '\n')
                wf.flush()
                os.fsync(wf.fileno()) # flush + fsync to try to be especially certain we write to checkpointing file before moving on
                print(f"\tDumped Whisper transcription to {output_jsonl_fpath}: {result}")
            else:
                print(f"Skipping transcription for audio file # {index+1}; previously completed {len(data)} transcriptions..")

    if get_llm_translations:
        llm_key = open("chat_gpt_secret_key.txt").readline().strip()
        client = OpenAI(
                            api_key=llm_key
                        )
        llm_instruction_msg = f"In each message received, attempt to remove or correct any audio transcription artifacts, then translate the message from Japanese to English, and return the OCR-corrected Japanese, followed by its English translation, and, lastly, append readings for any rare Japanese words."
        output_jsonl_fpath_w_llm_trans = os.path.splitext(output_jsonl_fpath)[0] + "_wLLMTranslations.jsonl"

        # NOTE our initial application involves running this on a highly unstable old laptop, so it is critical to checkpoint/restart from checkpoints
        assert os.path.isfile(output_jsonl_fpath)
        with open(output_jsonl_fpath, encoding="utf-8") as rf:
            notrans_data = [json.loads(line) for line in rf.readlines()]
        trans_data = []
        if os.path.isfile(output_jsonl_fpath_w_llm_trans):
            with open(output_jsonl_fpath_w_llm_trans, encoding="utf-8", mode='r') as rf:   
                trans_data = [json.loads(line) for line in rf.readlines()]

        with open(output_jsonl_fpath_w_llm_trans, encoding="utf-8", mode='a') as wf:
            for index, notrans_dict in enumerate(notrans_data):
                if index >= len(trans_data) - 1:
                    assert "llm_translation" not in notrans_dict.keys()
                    audio_text = notrans_dict["text"]
                    try:
                        response = client.responses.create(
                                                            model="gpt-5.2",
                                                            instructions=llm_instruction_msg,
                                                            input=audio_text,
                                                            )
                        r = response.output_text
                        u = response.usage
                        print(f"For {notrans_dict}, translated\n\t{audio_text}\n\t\tto\n\t{r}")
                        print(f"LLM token (etc) usage: {u}")
                        notrans_dict["llm_translation"] = r
                    except Exception as e:
                        print(f"Translation of {audio_text} failed with error: {e}")
                        print(f"Setting translation to |~|NOT AVAILABLE|~|...")
                        notrans_dict["llm_translation"] = "|~|NOT AVAILABLE|~|"
     
                    wf.write(json.dumps(notrans_dict, ensure_ascii=False) + '\n')
                    wf.flush()
                    os.fsync(wf.fileno()) # flush + fsync to try to be especially certain we write to checkpointing file before moving on
                    print(f"\tAppended LLM translation and dumped transcription+translation to {output_jsonl_fpath_w_llm_trans}: {notrans_dict}")

if __name__ == "__main__":
    # NOTE Example run command:
    # python transcribe_audio_w_translations.py transcribeAudioFiles --audio_fpaths_input_glob_expr="audio_files/Mixed_Exam_Audio_25_08_2026/*.mp3" --output_jsonl_fpath="mixed_exam_audio_whisper_test.jsonl" --test_len=3
    fire.Fire()
