# -*- coding: utf-8 -*-
import json
import wave
import codecs

def get_duration_wave(w):
    audio = wave.open(w)
    duration = float(audio.getnframes()) / audio.getframerate()
    audio.close()
    return duration

def aishell_filter():
    _data_path = "/mnt/cephfs2/asr/users/fanlu/data"
    lines = open(_data_path + "/aishell_label_dict.txt").readlines()
    for line in lines:
        s = set(line.strip().split())
    out_file = codecs.open(_data_path + "/aishell_validation2.json", 'w', encoding="utf-8")
    out_file2 = codecs.open(_data_path + "/aishell_test2.json", 'w', encoding="utf-8")
    for line in open(_data_path + "/aishell_validation.json").readlines():
        dic = json.loads(line)
        text = dic.get("text").split()
        cc = False
        for t in text:
            if t not in s:
                cc = True
                break
        if cc:
            continue
        else:
            out_file.write(line)



def ai_2_word():
    _data_path="/opt/cephfs1/asr/users/fanlu/data/"
    lines = open(_data_path + "data_aishell/transcript/aishell_transcript_v0.8.txt").readlines()
    out_file = codecs.open(_data_path + "/aishell_train.json", 'w', encoding="utf-8")
    out_file1 = codecs.open(_data_path + "/aishell_validation.json", 'w', encoding="utf-8")
    out_file2 = codecs.open(_data_path + "/aishell_test.json", 'w', encoding="utf-8")
    for line in lines:
        rs = line.strip().split(" ")
        # import pdb
        # pdb.set_trace()
        ps = generate_zi_label("".join(rs[1:]))

        if rs[0][6:11] <= "S0723":
            wav = _data_path + "data_aishell/wav/train/" + rs[0][6:11] + "/" + rs[0] + ".wav"
            # dir = _data_path + "data_aishell/wav/train_aug/" + rs[0][6:11] + "/" + rs[0]
            # for w in glob.glob(dir + "*.wav"):
            duration = get_duration_wave(wav)
            line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
            out_file.write(line + "\n")
        elif rs[0][6:11] <= "S0763":
            wav = _data_path + "data_aishell/wav/dev/" + rs[0][6:11] + "/" + rs[0] + ".wav"
            duration = get_duration_wave(wav)
            line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
            out_file1.write(line + "\n")
        else:
            wav = _data_path + "data_aishell/wav/test/" + rs[0][6:11] + "/" + rs[0] + ".wav"
            duration = get_duration_wave(wav)
            line = "{\"key\":\"" + wav + "\", \"duration\": " + str(duration) + ", \"text\":\"" + " ".join(ps) + "\"}"
            out_file2.write(line + "\n")
    out_file.close()
    out_file1.close()
    out_file2.close()

def generate_zi_label(label):
    # from create_desc_json import english_word
    try:
        str_ = label.strip().decode('utf-8')
    except:
        str_ = label.strip()
    l = []
    for ch in str_:
        if ch != u' ':
            l.append(ch)
    return l

if __name__ == "__main__":
    #ai_2_word()
    aishell_filter()
