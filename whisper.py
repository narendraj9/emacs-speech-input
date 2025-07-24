#!/usr/bin/env python

import io
import os
import threading
import wave
import pyaudio
from openai import OpenAI


class WhisperTranscriber:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.record_seconds = 3  # Process audio in 3-second chunks
        self.audio = pyaudio.PyAudio()
        self.running = False

    def record_chunk(self):
        """Record a chunk of audio"""
        stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        frames = []
        for _ in range(
            0, int(self.sample_rate / self.chunk_size * self.record_seconds)
        ):
            data = stream.read(self.chunk_size)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        return b"".join(frames)

    def audio_to_wav_bytes(self, audio_data):
        """Convert raw audio data to WAV bytes"""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.audio.get_sample_size(self.audio_format))
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data)

        wav_buffer.seek(0)
        return wav_buffer

    def transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper API"""
        try:
            wav_bytes = self.audio_to_wav_bytes(audio_data)
            wav_bytes.name = "audio.wav"  # Required by the API

            response = self.client.audio.transcriptions.create(
                model="whisper-1", file=wav_bytes, language="en"
            )

            return response.text.strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def start_continuous_transcription(self):
        """Start continuous transcription"""
        print("Starting continuous transcription. Press Enter to stop...\n")
        self.running = True

        def transcription_loop():
            while self.running:
                # Record audio chunk
                audio_data = self.record_chunk()

                # Transcribe it
                transcript = self.transcribe_audio(audio_data)

                if transcript:
                    print(f"Transcription: {transcript}")

        # Start transcription in separate thread
        transcription_thread = threading.Thread(target=transcription_loop)
        transcription_thread.daemon = True
        transcription_thread.start()

        # Wait for user input
        input()

        print("Stopping transcription...")
        self.running = False
        transcription_thread.join(timeout=5)

    def cleanup(self):
        """Clean up audio resources"""
        self.audio.terminate()


def main():
    transcriber = WhisperTranscriber()

    try:
        transcriber.start_continuous_transcription()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        transcriber.cleanup()


if __name__ == "__main__":
    main()
