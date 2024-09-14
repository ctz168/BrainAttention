import pyaudio
import sounddevice as sd

import torch
import torch.nn as nn
import numpy as np
import threading
import cv2
import os
import shutil
import gc
import wave
import time
from moviepy.editor import VideoFileClip

class STDP:
    def __init__(self, input_size, num_neurons, learning_rate=0.01, tau_pos=20.0, tau_neg=20.0):
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.learning_rate = learning_rate
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg

    def update_weights(self, pre_spikes, post_spikes, weights):

        # Compute spike differences
        spike_diff = post_spikes.unsqueeze(1) - pre_spikes.unsqueeze(2)  # Adjust dimensions for broadcasting

        # Compute delta_w
        delta_w = self.learning_rate * (torch.exp(-torch.abs(spike_diff) / self.tau_pos) -torch.exp(-torch.abs(spike_diff) / self.tau_neg))

        # Sum or mean across the post_spikes and pre_spikes dimensions
        delta_w = delta_w.sum(dim=0)  # Average over the batch dimension/优化空间

        new_weights = weights + delta_w
        return new_weights

class STDPUpdateRule:
    def __init__(self, tau_pos, tau_neg, learning_rate):
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.learning_rate = learning_rate

    def update_weights_delta(self, pre_activations, post_activations):
        # 计算权重更新值
        spike_diff = post_activations.unsqueeze(1) - pre_activations.unsqueeze(2)
        delta_w = self.learning_rate * (torch.exp(-torch.abs(spike_diff) / self.tau_pos) - torch.exp(-torch.abs(spike_diff) / self.tau_neg))
        # 应用权重更新
        delta_w = delta_w.mean(dim=0)  # Average over the batch dimension

        return delta_w
class ComplexNeuron(nn.Module):
    def __init__(self, input_size, num_neurons, sparsity=0.1):
        super(ComplexNeuron, self).__init__()
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.sparsity = sparsity

        # Initialize the connections matrix
        self.connections = nn.Parameter(torch.randn(input_size, num_neurons) * sparsity)

    def forward(self, x):
        # Implement the forward pass
        out = torch.mm(x, self.connections)
        return out


class STDPAttentionMultiModal(nn.Module):
    def __init__(self, av_embed_size, heads, num_neurons, sparsity=0.1):
        super(STDPAttentionMultiModal, self).__init__()
        self.av_attention = nn.MultiheadAttention(embed_dim=av_embed_size, num_heads=heads)
        self.complex_neuron_av = ComplexNeuron(input_size=av_embed_size, num_neurons=num_neurons, sparsity=sparsity)
        self.stdp_av = STDP(av_embed_size, num_neurons)
        self.learning_rate = 0.01  # 手动更新学习率
        self.stdp_update_rule = STDPUpdateRule(tau_pos=20.0, tau_neg=20.0, learning_rate=self.learning_rate)


    def forward(self, av_value, av_key, av_query,mask=None):
        # Process audio
        av_attention_out,_ = self.av_attention(av_query, av_key, av_value, key_padding_mask=mask)
        av_attention_out = av_attention_out.squeeze(0)
        av_neuron_out = self.complex_neuron_av(av_attention_out)

        if isUpdateWeights:
            pre_spikes_av = av_attention_out
            post_spikes_av = av_neuron_out
            self.complex_neuron_av.connections.data = self.stdp_av.update_weights(pre_spikes_av, post_spikes_av,self.complex_neuron_av.connections.data)
            # 手动计算注意力权重的更新
            pre_activations = av_value.squeeze(0)
            post_activations = av_attention_out
            param_list = list(self.av_attention.parameters())
            for idx, param in enumerate(param_list):
                w_delta=self.stdp_update_rule.update_weights_delta(pre_activations, post_activations)
                if idx==0:
                    w_delta = w_delta.repeat(3, 1)
                    param.data+=w_delta # 直接更新参数数据
                elif idx==2:
                    param.data +=w_delta
        audio_output = av_neuron_out[:, :audioblocksize]

        return audio_output  # Only return audio output for playback


# Audio and Video handling
class AudioHandler:
    def __init__(self):
        self.chunk = audioblocksize  # Buffer size
        self.sample_rate = 16000
        self.p = pyaudio.PyAudio()

        self.stream_in = self.p.open(format=pyaudio.paInt16,
                                     channels=1,
                                     rate=self.sample_rate,
                                     input=True,
                                     frames_per_buffer=self.chunk)

        self.stream_out = self.p.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=self.sample_rate,
                                      output=True,
                                      frames_per_buffer=self.chunk)

    def read_audio(self):
        try:
            audio_data=np.frombuffer(self.stream_in.read(self.chunk, exception_on_overflow=False), dtype=np.int16)
            # if not os.path.isfile("test.wav"):
            #     with wave.open('test.wav', 'w') as wf:
            #         # 设置 WAV 文件的参数
            #         wf.setnchannels(1)
            #         wf.setsampwidth(2)
            #         wf.setframerate(16000)
            #         # 写入音频数据
            #         wf.writeframes(audio_data.tobytes())
            # else:
            #     with wave.open('test.wav', 'rb') as wf:
            #         existing_audio_data =wf.readframes(wf.getnframes())
            #     with wave.open('test.wav', 'w') as wf:
            #         # 设置 WAV 文件的参数
            #         wf.setnchannels(1)
            #         wf.setsampwidth(2)
            #         wf.setframerate(16000)
            #         # 写入音频数据
            #
            #         wf.writeframes(existing_audio_data+audio_data.tobytes())

            return audio_data
        except IOError as e:
            print(f"Error reading audio: {e}")
            return np.zeros(self.chunk, dtype=np.int16)

    def write_audio(self, audio_data):
        self.stream_out.write(audio_data.astype(np.int16).tobytes())
        current_time = time.time()
        readable_time = time.ctime(current_time)
        print("当前时间是:", readable_time)
    def close(self):
        self.stream_in.stop_stream()
        self.stream_in.close()
        self.stream_out.stop_stream()
        self.stream_out.close()
        self.p.terminate()


class VideoHandler:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Open the first camera

    def read_video(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error reading video frame.")
            return None
        return frame

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
class FakeVideoHandler:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Open the first camera

    def read_video(self):

        return  np.zeros((raw_video_width,raw_video_height,3))


    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

class FileAVHandler():

    def __init__(self,video_path):
        # 读取视频文件
        self.video_clip = VideoFileClip(video_path)
        self.duration = self.video_clip.duration  # 视频总时长
        self.totalframe = int(self.duration * self.video_clip.fps)
        self.framecount=0
        self.last_output_audio=None
        self.this_file_end = False
        # 音频参数
        # samplerate = video_clip.audio.fps  # 音频采样率通常与视频帧率相同
        # 初始化音频处理
        # myRecorder = sd.Recorder(samplerate=samplerate, channels=1)  # 单声道录音器
        # myRecorder.initialize()
        # 逐帧处理视频
        # 计算总帧数
        # 停止录音器（如果之前开启了实时播放）
        # myRecorder.stop()
        # 释放资源video_clip.close()


    def get_Video(self):

        return self.video_clip.get_frame(self.framecount)

    def get_Audio(self):

        # 同步提取音频
        start_time = self.framecount / self.video_clip.fps  # 当前帧对应的时间点
        end_time = start_time + (1 / self.video_clip.fps)  # 音频块的结束时间点
        # 直接从audio_clip获取音频数据

        if end_time<self.video_clip.duration:
            audio_clip = self.video_clip.audio.subclip(start_time, end_time)
        else:
            self.this_file_end = True
            return None

        audio_data = audio_clip.to_soundarray(fps=44100)
        # 获取声道数（moviepy默认处理为立体声，所以通常是2）
        # 如果音频是双通道的，将其转换为单通道
        audio_data = audio_data[:, 0] if audio_data.ndim > 1 else audio_data

        if self.last_output_audio is not None:

            # 计算第一段音频的RMS值
            rms_audio1 = np.sqrt(np.mean(audio_data ** 2))
            # 将第二段音频的RMS值限制在第一段音频RMS的比例范围内
            max_ratio = 0.3  # 第二段音频最大不超过第一段音频的30%
            rms_audio2 = np.sqrt(np.mean(self.last_output_audio ** 2))
            scaling_factor = (rms_audio1 * max_ratio) / rms_audio2
            minlength = min(audio_data.shape[0], self.last_output_audio.shape[0])
            delaysamples=160
            padding_audio=np.pad(self.last_output_audio* scaling_factor,(delaysamples,0),mode='constant')
            # 根据scaling_factor调整第二段音频的音量
            audio_data=audio_data[0:minlength]+padding_audio[0:minlength]

        audio_data = audio_data[0:audioblocksize]
        if isRecordAudio:
            if not os.path.isfile("mergeaudio.wav"):
                int_audio_array = (audio_data * 32767).astype(np.int16)
                with wave.open('mergeaudio.wav', 'w') as wf:
                    # 设置 WAV 文件的参数
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(44100)
                    # 写入音频数据
                    wf.writeframes(int_audio_array.tobytes())
            else:
                with wave.open('mergeaudio.wav', 'rb') as wf:
                    existing_audio_data = wf.readframes(wf.getnframes())
                with wave.open('mergeaudio.wav', 'w') as wf:
                    # 设置 WAV 文件的参数
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(44100)
                    # 写入音频数据
                    int_audio_array = (audio_data * 32767).astype(np.int16)
                    wf.writeframes(existing_audio_data + int_audio_array.tobytes())
                    sd.play(audio_data, 44100)
                    sd.wait()
        audio_data = torch.tensor(audio_data, dtype=torch.float)
        audio_data = audio_data.clone().detach().float()


        return audio_data
def processing_train_by_file(model):
    for root, dirs, files in os.walk("video/"):
        for file in files:
            # 检查文件扩展名是否为常见的视频文件格式
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                # 打印视频文件的完整路径
                train_file = os.path.join(root, file)
                file_av_handler=FileAVHandler(train_file)

                while True:
                    print("训练：", file_av_handler.framecount)

                    audio_np=file_av_handler.get_Audio()
                    if audio_np is None:
                        break
                    else:
                        audio_tensor =torch.tensor(audio_np).unsqueeze(0).to(device)

                    video_frame=file_av_handler.get_Video()
                    video_frame = cv2.resize(video_frame, (raw_video_width, raw_video_height))
                    if isRecordAudio:
                        cv2.imshow('GenerateVideo', video_frame)
                        cv2.waitKey(1)
                    # video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
                    video_tensor = torch.tensor(video_frame.reshape(1, raw_video_width * raw_video_height * 3),
                                                dtype=torch.float32).to(device)


                    av_tensor = torch.cat((audio_tensor, video_tensor), dim=1).unsqueeze(0).to(device)
                    processed_audio = model(av_tensor, av_tensor, av_tensor)
                    # Ensure processed_audio is 1D for playback
                    processed_audio=processed_audio.to('cpu')
                    processed_audio = processed_audio.squeeze().detach().numpy()

                    file_av_handler.framecount+=1
                    if file_av_handler.framecount>0 :
                        file_av_handler.last_output_audio=processed_audio

                    if file_av_handler.framecount>file_av_handler.totalframe or file_av_handler.this_file_end:
                        break
                torch.save(model.state_dict(), weight_file)
                move_video_to_trained_folder(train_file)
                file_av_handler.video_clip.close()
                gc.collect()
    print("no more video!")
    exit()
# Real-time processing loop
def processing_loop(model, audio_handler, video_handler):
    while True:
        audio_input = audio_handler.read_audio()
        audio_tensor = torch.tensor(audio_input.copy(), dtype=torch.float32).unsqueeze(0).to(device)

        if audio_tensor.size(-1) != audioblocksize:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, audioblocksize - audio_tensor.size(-1))).to(device)

        video_frame = video_handler.read_video()
        video_frame = cv2.resize(video_frame, ( raw_video_width,raw_video_height))
        # video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        if video_frame is not None:
            # Convert video frame to tensor and reshape

            video_tensor = torch.tensor(video_frame.reshape(1,raw_video_width*raw_video_height*3), dtype=torch.float32).to(device)  # Assuming embed_dim is also 512
            av_tensor=torch.cat((audio_tensor,video_tensor),dim=1).unsqueeze(0).to(device)
            # Model processing

            processed_audio = model(av_tensor, av_tensor, av_tensor)
            # Ensure processed_audio is 1D for playback
            processed_audio = processed_audio.squeeze().detach().numpy()

            audio_handler.write_audio(processed_audio)

def move_video_to_trained_folder(source_file_path):
    trainedfolder="video/trained/"
    if not os.path.exists(trainedfolder):
        os.makedirs(trainedfolder)
    # 定义目标文件的完整路径
    destination_file_path = os.path.join(trainedfolder, os.path.basename(source_file_path))
    # 移动文件
    shutil.move(source_file_path, destination_file_path)

def main():

    model = STDPAttentionMultiModal(av_embed_size=totoal_input_size, heads=8, num_neurons=10000, sparsity=0.1)
    model.to(device)
    # Load weights if they exist

    if os.path.exists(weight_file):
        # model.load_state_dict(torch.load(weight_file))
        # 使用 map_location 将存储映射到 CPU
        model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        print(f"Loaded weights from {weight_file}.")
    else:
        print(f"File {weight_file} does not exist. Starting with random weights.")
    if isTrainByFile:
         processing_train_by_file(model)
    else:
        audio_handler = AudioHandler()
        video_handler = FakeVideoHandler()

        threading.Thread(target=processing_loop, args=(model, audio_handler, video_handler)).start()

        try:
            # Keep the main thread alive
            while True:
                pass
        except KeyboardInterrupt:
            print("Terminating...")
        finally:
            # Save weights
            torch.save(model.state_dict(), weight_file)
            audio_handler.close()
            video_handler.close()

weight_file = 'mojo_weights.pth'
isTrainByFile=True
isUpdateWeights=True
isRecordAudio=False
raw_video_width=64
raw_video_height=36
audioblocksize=1024
totoal_input_size=raw_video_width*raw_video_height*3+audioblocksize
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
if __name__ == "__main__":
    main()