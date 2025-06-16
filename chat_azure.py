import argparse
import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import CancelFrame, Frame, StartFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)
from pipecat.services.azure.llm import AzureLLMService
from pipecat.utils.asyncio import TaskManager

# Load environment variables
load_dotenv()
if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_DEPLOYMENT"):
    logger.error("Missing AZURE_OPENAI_API_KEY or AZURE_OPENAI_DEPLOYMENT in .env")
    exit(1)

# Configure AzureLLMService
llm = AzureLLMService(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
)

# Pricing constants (INR per token, 1 USD = 83 INR)
INPUT_TOKEN_COST_INR = 0.0002075  # $2.50 per million tokens
OUTPUT_TOKEN_COST_INR = 0.00083  # $10.00 per million tokens


# Terminal input processor
class TerminalInputProcessor(FrameProcessor):
    def __init__(self, llm_service, deployment, **kwargs):
        super().__init__(**kwargs)
        self._started = False
        self._llm_service = llm_service
        self._deployment = deployment
        self._context = OpenAILLMContext(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Keep responses Gen Z, lit, and short.",
                }
            ]
        )
        self._cancelled = False
        self._response_queue = asyncio.Queue()
        self._task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, StartFrame):
            self._started = True
            await super().process_frame(frame, direction)
            self._task = asyncio.create_task(self._input_loop())
            return
        if isinstance(frame, CancelFrame):
            self._cancelled = True
            if self._task:
                self._task.cancel()
            return
        if not self._started:
            logger.debug("Waiting for StartFrame")
            return
        await super().process_frame(frame, direction)

    async def _input_loop(self):
        while not self._cancelled:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() == "exit":
                    logger.info("User requested exit")
                    self._cancelled = True
                    await self.push_frame(CancelFrame())
                    break
                if not user_input:
                    continue
                self._context.add_message({"role": "user", "content": user_input})
                response = await self._llm_service._client.chat.completions.create(
                    model=self._deployment,
                    messages=self._context.get_messages(),
                )
                response_text = response.choices[0].message.content
                self._context.add_message({"role": "assistant", "content": response_text})
                frame = TextFrame(response_text)
                frame.metadata = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                await self.push_frame(frame)
                await self._response_queue.get()
            except asyncio.CancelledError:
                self._cancelled = True
                break
            except KeyboardInterrupt:
                logger.info("User interrupted input")
                self._cancelled = True
                await self.push_frame(CancelFrame())
                break
            except Exception as e:
                logger.error(f"LLM error: {e}")


# Terminal output processor
class TerminalOutputProcessor(FrameProcessor):
    def __init__(self, input_processor, **kwargs):
        super().__init__(**kwargs)
        self._input_processor = input_processor

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        self._FrameProcessor__create_input_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TextFrame):
            print(f"Assistant: {frame.text}")
            if frame.metadata and "prompt_tokens" in frame.metadata:
                prompt_tokens = frame.metadata["prompt_tokens"]
                completion_tokens = frame.metadata["completion_tokens"]
                total_tokens = frame.metadata["total_tokens"]
                # Calculate cost in INR
                input_cost = prompt_tokens * INPUT_TOKEN_COST_INR
                output_cost = completion_tokens * OUTPUT_TOKEN_COST_INR
                total_cost = input_cost + output_cost
                # Format to four decimal places
                print(
                    f"Token Usage: Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}"
                )
                print(f"Cost: ₹{total_cost:.4f}")
            await self._input_processor._response_queue.put(None)
        await super().process_frame(frame, direction)


# Set up pipeline
pipeline = Pipeline(
    [
        input_processor := TerminalInputProcessor(
            llm_service=llm, deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
        ),
        TerminalOutputProcessor(input_processor=input_processor),
    ]
)


# Create and configure task
async def run_pipeline():
    task_manager = TaskManager()
    task_manager.set_event_loop(asyncio.get_event_loop())
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True, enable_metrics=True, enable_usage_metrics=True
        ),
        task_manager=task_manager,
    )
    try:
        await asyncio.sleep(0.1)
        await task.run()
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
        await task.cancel()
    finally:
        try:
            await asyncio.wait_for(task.cleanup(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Cleanup timed out")


# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run Pipecat pipeline with Azure LLM")
    return parser.parse_args()


# Main entry point
if __name__ == "__main__":
    parse_args()
    asyncio.run(run_pipeline())
