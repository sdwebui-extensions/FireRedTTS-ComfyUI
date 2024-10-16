import os,sys
import os.path as osp
now_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(now_dir)
tmp_dir = osp.join(now_dir, "tmp")
import re
import torch
import ffmpeg
import shutil
import tempfile
import torchaudio
import folder_paths
import numpy as np
from tqdm import tqdm
from comfy.utils import ProgressBar
from pydub import AudioSegment
from pydub.silence import split_on_silence
from huggingface_hub import snapshot_download
from fireredtts.fireredtts import FireRedTTS
from zhon.hanzi import punctuation
from zh_normalization import text_normalize
from LangSegment import LangSegment
LangSegment.setfilters(["zh", "en"])

SPLIT_WORDS = [
    "but", "however", "nevertheless", "yet", "still",
    "therefore", "thus", "hence", "consequently",
    "moreover", "furthermore", "additionally",
    "meanwhile", "alternatively", "otherwise",
    "namely", "specifically", "for example", "such as",
    "in fact", "indeed", "notably",
    "in contrast", "on the other hand", "conversely",
    "in conclusion", "to summarize", "finally"
]
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
aifsh_models = osp.join(folder_paths.models_dir,"AIFSH")
fireredtss_dir = osp.join(aifsh_models,"FireRedTTS")

def speed_change(input_audio, speed, sr):
    # 检查输入数据类型和声道数
    if input_audio.dtype != np.int16:
        raise ValueError("输入音频数据类型必须为 np.int16")


    # 转换为字节流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)

    # 变速处理
    output_stream = input_stream.filter('atempo', speed)

    # 输出流到管道
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio

class FireRedTTSNode:
    def __init__(self):
        if not osp.exists(osp.join(fireredtss_dir,"fireredtts_gpt.pt")):
            snapshot_download(repo_id="fireredteam/FireRedTTS",local_dir=fireredtss_dir)
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required":{
                "text":("TEXT",),
                "prompt_wav":("AUDIO",),
                "remove_slience":("BOOLEAN",{
                    "default": True
                }),
                "speed":("FLOAT",{
                    "default":1.0,
                    "min":0.5,
                    "max":2.0,
                    "step":0.05,
                    "round":0.001,
                    "display":"slider"
                }),
                "split_words":("STRING",{
                    "default":",".join(SPLIT_WORDS),
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip":"Enter custom words to split on, separated by commas. Leave blank to use default list.",
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_audio"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_FireRedTSS"

    def gen_audio(self,text,prompt_wav,remove_slience,speed,split_words):
        os.makedirs(tmp_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav",dir=tmp_dir) as f:
            waveform = prompt_wav["waveform"].squeeze(0)
            torchaudio.save(f.name,waveform,prompt_wav["sample_rate"])
            if remove_slience:
                combined = slice(f.name)
                combined.export(f.name, format="wav")
            prompt_wav = f.name
        # Split the input text into batches
        
        if len(text.encode('utf-8')) == len(text):
            max_chars = 400-len(text.encode('utf-8'))
        else:
            max_chars = 300-len(text.encode('utf-8'))
        
        if not split_words.strip():
            custom_words = [word.strip() for word in split_words.split(',')]
            global SPLIT_WORDS
            SPLIT_WORDS = custom_words
        gen_text_batches = split_text_into_batches(text, max_chars=max_chars)
        gen_text_batches = text_list_normalize(gen_text_batches)
        comfy_par = ProgressBar(len(gen_text_batches))
        tts = FireRedTTS(config_path=osp.join(now_dir,"config_24k.json"),
                         pretrained_path=fireredtss_dir,device=device)
        rec_wavs_list = []
        for i,i_text in enumerate(tqdm(gen_text_batches,total=len(gen_text_batches),desc="TTS ...")):
            '''
            for dot in punctuation:
                i_text = i_text.replace(dot,"")
            '''
            if i_text[-1] not in ['。', '.', '!', '！', '?', '？']:
                i_text += '.'
            print(f"sentence {i+1}, synthesize text:{i_text}")
            rec_wavs = tts.synthesize(prompt_wav=prompt_wav,text=i_text)
            rec_wavs = rec_wavs.detach().cpu().numpy()
            rec_wavs_list.append(rec_wavs)
            comfy_par.update(1)
        res_np = np.concatenate(rec_wavs_list,axis=1)
        if speed > 1.0 or speed < 1.0:
            res_np = res_np * 32768
            res_np = res_np.astype(np.int16)
            res_np = speed_change(res_np,speed,sr=24000)
            waveform = torch.from_numpy(res_np/32768).unsqueeze(0).unsqueeze(0)
        else:
            waveform = torch.from_numpy(res_np).unsqueeze(0)
        print(waveform.shape)
        res_audio = {
            "waveform": waveform,
            "sample_rate": 24000
        }
        shutil.rmtree(tmp_dir)
        return (res_audio, )

def text_list_normalize(texts):
    text_list = []
    for text in texts:
        for tmp in LangSegment.getTexts(text):
            normalize = text_normalize(tmp.get("text"))
            if normalize != "" and tmp.get("lang") == "en" and normalize not in ["."]:
                if len(text_list) > 0:
                    text_list[-1] += normalize
                else:
                    text_list.append(normalize)
            elif tmp.get("lang") == "zh":
                text_list.append(normalize)
            else:
                if len(text_list) > 0:
                    text_list[-1] += tmp.get("text")
                else:
                    text_list.append(tmp.get("text"))
    return text_list

def split_text_into_batches(text, max_chars=200, split_words=SPLIT_WORDS):
    if len(text.encode('utf-8')) <= max_chars:
        return [text]
    if text[-1] not in ['。', '.', '!', '！', '?', '？']:
        text += '.'
        
    sentences = re.split('([。.!?！？])', text)
    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]
    
    batches = []
    current_batch = ""
    
    def split_by_words(text):
        words = text.split()
        current_word_part = ""
        word_batches = []
        for word in words:
            if len(current_word_part.encode('utf-8')) + len(word.encode('utf-8')) + 1 <= max_chars:
                current_word_part += word + ' '
            else:
                if current_word_part:
                    # Try to find a suitable split word
                    for split_word in split_words:
                        split_index = current_word_part.rfind(' ' + split_word + ' ')
                        if split_index != -1:
                            word_batches.append(current_word_part[:split_index].strip())
                            current_word_part = current_word_part[split_index:].strip() + ' '
                            break
                    else:
                        # If no suitable split word found, just append the current part
                        word_batches.append(current_word_part.strip())
                        current_word_part = ""
                current_word_part += word + ' '
        if current_word_part:
            word_batches.append(current_word_part.strip())
        return word_batches

    for sentence in sentences:
        if len(current_batch.encode('utf-8')) + len(sentence.encode('utf-8')) <= max_chars:
            current_batch += sentence
        else:
            # If adding this sentence would exceed the limit
            if current_batch:
                batches.append(current_batch)
                current_batch = ""
            
            # If the sentence itself is longer than max_chars, split it
            if len(sentence.encode('utf-8')) > max_chars:
                # First, try to split by colon
                colon_parts = sentence.split(':')
                if len(colon_parts) > 1:
                    for part in colon_parts:
                        if len(part.encode('utf-8')) <= max_chars:
                            batches.append(part)
                        else:
                            # If colon part is still too long, split by comma
                            comma_parts = re.split('[,，]', part)
                            if len(comma_parts) > 1:
                                current_comma_part = ""
                                for comma_part in comma_parts:
                                    if len(current_comma_part.encode('utf-8')) + len(comma_part.encode('utf-8')) <= max_chars:
                                        current_comma_part += comma_part + ','
                                    else:
                                        if current_comma_part:
                                            batches.append(current_comma_part.rstrip(','))
                                        current_comma_part = comma_part + ','
                                if current_comma_part:
                                    batches.append(current_comma_part.rstrip(','))
                            else:
                                # If no comma, split by words
                                batches.extend(split_by_words(part))
                else:
                    # If no colon, split by comma
                    comma_parts = re.split('[,，]', sentence)
                    if len(comma_parts) > 1:
                        current_comma_part = ""
                        for comma_part in comma_parts:
                            if len(current_comma_part.encode('utf-8')) + len(comma_part.encode('utf-8')) <= max_chars:
                                current_comma_part += comma_part + ','
                            else:
                                if current_comma_part:
                                    batches.append(current_comma_part.rstrip(','))
                                current_comma_part = comma_part + ','
                        if current_comma_part:
                            batches.append(current_comma_part.rstrip(','))
                    else:
                        # If no comma, split by words
                        batches.extend(split_by_words(sentence))
            else:
                current_batch = sentence
    
    if current_batch:
        batches.append(current_batch)
    
    return batches


def slice(audio_path):
    """_summary_

    Args:
        audio_path (_type_): audio path
    """
    try:
        audio = AudioSegment.from_file(audio_path)
    except:
        print(audio_path)
        return 0

    segments = split_on_silence(
        audio, min_silence_len=200, silence_thresh=-50, seek_step=100, keep_silence=100
    )

    print("---segments:\n", segments)

    combined = segments[0]
    for i in range(1, len(segments)):
        combined += segments[i]
    return combined

NODE_CLASS_MAPPINGS = {
    "FireRedTTSNode": FireRedTTSNode
}