#!/usr/bin/env python

import asyncio
import base64
import json
import os
import pyaudio
import websockets
from websockets.exceptions import ConnectionClosed


class OpenAIRealtimeTranscriber:
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError("API_KEY environment variable is required")

        self.websocket = None
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 24000  # OpenAI real-time API uses 24kHz
        self.chunk_size = 1024
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.running = False

    async def on_session_created(self, event):
        """Handle session creation - equivalent to on_metadata"""
        print(f"\n\nSession created: {json.dumps(event, indent=2)}\n\n")

    async def on_speech_started(self, event):
        """Handle speech detection start"""
        print(f"\n\nSpeech started: {json.dumps(event, indent=2)}\n\n")

    async def on_speech_stopped(self, event):
        """Handle speech detection stop"""
        print(f"Speech stopped: {json.dumps(event, indent=2)}")

    async def on_transcription_completed(self, event):
        """Handle completed transcription - equivalent to on_message"""
        print(f"Handling {event}")
        text = ""
        is_final = False
        if "transcript" in event:
            text = event["transcript"]
            is_final = True
        if "delta" in event:
            text = event["delta"]
        if text.strip():
            print(
                "Output: "
                + json.dumps(
                    {
                        "transcript": text,
                        "is_final": is_final,
                        "item_id": event["item_id"],
                    }
                )
            )

    async def on_error(self, event):
        """Handle errors"""
        print(f"Error: {json.dumps(event, indent=2)}")

    async def handle_server_events(self):
        """Handle incoming events from OpenAI real-time API"""
        try:
            async for message in self.websocket:
                try:
                    event = json.loads(message)
                    event_type = event.get("type", "")

                    if event_type == "session.created":
                        await self.on_session_created(event)
                    elif event_type == "input_audio_buffer.speech_started":
                        await self.on_speech_started(event)
                    elif event_type == "input_audio_buffer.speech_stopped":
                        await self.on_speech_stopped(event)
                    elif (
                        event_type
                        == "conversation.item.input_audio_transcription.delta"
                    ):
                        await self.on_transcription_completed(event)
                    elif (
                        event_type
                        == "conversation.item.input_audio_transcription.completed"
                    ):
                        await self.on_transcription_completed(event)
                    elif event_type == "error":
                        await self.on_error(event)
                    else:
                        pass
                        # Handle other events if needed
                        print(f"Received event: {event_type}")

                except json.JSONDecodeError as e:
                    print(f"Failed to decode message: {e}")
                except Exception as e:
                    print(f"Error handling server event: {e}")

        except ConnectionClosed:
            print("Connection to OpenAI closed")
        except Exception as e:
            print(f"Error in handle_server_events: {e}")

    async def send_session_config(self):
        """Configure the session for transcription"""
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "instructions": "You are a helpful assistant that transcribes audio.",
                "input_audio_format": "pcm16",
                "input_audio_transcription": {"model": "gpt-4o-transcribe"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
            },
        }

        await self.websocket.send(json.dumps(session_config))

    def start_audio_capture(self):
        """Start capturing audio from microphone"""
        self.stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

    async def stream_audio(self):
        """Capture and stream audio to OpenAI"""
        self.start_audio_capture()

        try:
            while self.running:
                try:
                    # Read audio data
                    audio_data = self.stream.read(
                        self.chunk_size, exception_on_overflow=False
                    )

                    # Encode audio data as base64
                    audio_base64 = base64.b64encode(audio_data).decode("utf-8")

                    # Send audio data to OpenAI
                    audio_event = {
                        "type": "input_audio_buffer.append",
                        "audio": audio_base64,
                    }

                    await self.websocket.send(json.dumps(audio_event))

                    # Small delay to prevent overwhelming the API
                    await asyncio.sleep(0.01)

                except Exception as e:
                    print(f"Error streaming audio: {e}")
                    break

        except Exception as e:
            print(f"Error in audio streaming: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()

    async def connect_and_run(self):
        """Main connection and run loop"""
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            async with websockets.connect(url, additional_headers=headers) as websocket:
                self.websocket = websocket

                # Configure the session
                await self.send_session_config()

                self.running = True

                # Start audio streaming and event handling concurrently
                audio_task = asyncio.create_task(self.stream_audio())
                events_task = asyncio.create_task(self.handle_server_events())

                # Wait for user input in a non-blocking way
                def check_input():
                    try:
                        input("Press Enter to stop recording...\n\n")
                        return True
                    except:
                        return False

                # Run until user presses Enter
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, check_input)

                self.running = False

                # Cancel tasks
                audio_task.cancel()
                events_task.cancel()

                try:
                    await audio_task
                except asyncio.CancelledError:
                    pass

                try:
                    await events_task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            if self.audio:
                self.audio.terminate()


async def main():
    """Main function"""
    try:
        transcriber = OpenAIRealtimeTranscriber()
        await transcriber.connect_and_run()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
