import os
import re
import time
import logging
import io
from typing import Dict, List, Union

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Poll
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler,
)
import google.generativeai as genai
from pdf2image import convert_from_path
from PIL import Image

# --- Configuration and Setup ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Config:
    """Configuration for the Gemini API and bot behavior."""

    def __init__(self):
        self.api_key: str = "AIzaSyBoMJG8yYLaVPpagG7BfrirzYWtX26sjoE"  # Hardcoded, but changeable
        self.model_name: str = "models/gemini-2.0-pro-exp"  # Default model
        self.chunk_size: int = 5000  # Maximum size of text chunks for Gemini
        self.max_tokens: int = 8000  # Maximum generation length.
        self.model = None  # Gemini model instance
        self.fixed_ocr_model: str = "models/gemini-2.0-flash-lite"  # Model used ONLY for OCR
        self.mode: str = "mcq"  # Default processing mode: 'mcq', 'qa', or 'custom'
        self.custom_prompt: str = ""  # Store custom prompt
        self.pro_model_cooldown = 30  # Cooldown for pro models
        self.initialize_model()

        # Define prompts as dictionaries for easier management and extension
        self.prompts = {
            "mcq": (
                "From the following text, create exam-oriented multiple-choice questions (MCQs) in Hindi with explanations:\n"
                "- Cover the entire content comprehensively without adding or removing information.\n"
                "- Focus on key facts like names, years, and terms to make it exam-ready.\n"
                "- Format each MCQ as:\n"
                "  Q: [Question]\n"
                "  A: [Option A]\n"
                "  B: [Option B]\n"
                "  C: [Option C]\n"
                "  D: [Option D]\n"
                "  Correct: [A/B/C/D]\n"
                "  E: [Explanation]\n"
                "- Keep it concise and in Hindi.\n"
                "Text:\n{text}"
            ),
            "qa": (
                "From the following text, create a set of simple quiz questions and answers in Hindi with explanations:\n"
                "- Focus on key facts like names, years, and terms.\n"
                "- Format each pair as: **Q:** [Question] | **A:** [Answer] | **E:** [Explanation]\n"
                "- Keep it concise and in Hindi.\n"
                "Text:\n{text}"
            ),
        }

    def initialize_model(self):
        """Initializes the Gemini model (called during __init__ and set_api_key)."""
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                logger.info(f"Gemini model {self.model_name} initialized.")
            except Exception as e:
                logger.error(f"Error initializing Gemini model: {e}")
                self.model = None  # Ensure model is None on failure
        else:
            logger.warning("API key not set.  Gemini model not initialized.")
            self.model = None


    def set_api_key(self, api_key: str):
        """Sets the API key and re-initializes the Gemini model."""
        self.api_key = api_key
        self.initialize_model()


    def set_model(self, model_name: str):
        """Changes the Gemini model used for generation."""
        self.model_name = model_name
        self.initialize_model()  # Re-initialize with the new model


    def set_mode(self, mode: str):
        """Sets the processing mode ('mcq', 'qa', 'custom')."""
        if mode.lower() in ["mcq", "qa", "custom"]:
            self.mode = mode.lower()
        else:
            raise ValueError("Invalid mode.  Choose 'mcq', 'qa', or 'custom'.")

    def set_custom_prompt(self, custom_prompt):
        self.custom_prompt = custom_prompt
        self.mode = "custom"

    def get_prompt(self, text: str) -> str:
        """Retrieves the appropriate prompt based on the current mode."""

        if self.mode == "custom":
            return self.custom_prompt + "\nText:\n{text}".format(text=text)
        elif self.mode in self.prompts:
            return self.prompts[self.mode].format(text=text)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")  # Should not happen, but good practice

    def is_pro_model(self) -> bool:
        """Checks if the current model is a 'pro' model requiring a cooldown."""
        pro_models = [
            "gemini-1.0-pro",
            "gemini-1.5-pro-latest",
            "models/gemini-2.0-pro-exp",
            "models/gemini-2.0-pro-exp-02-05",
        ]
        return self.model_name in pro_models

config = Config()

AVAILABLE_MODELS = [
    {"full": "gemini-1.5-pro-latest", "short": "1.5-pro-latest"},
    {"full": "gemini-1.0-pro", "short": "1.0-pro"},
    {"full": "gemini-1.5-flash", "short": "1.5-flash"},
    {"full": "models/gemini-2.0-flash-exp", "short": "2.0-flash-exp"},
    {"full": "models/gemini-2.0-flash", "short": "2.0-flash"},
    {"full": "models/gemini-2.0-flash-001", "short": "2.0-flash-001"},
    {"full": "models/gemini-2.0-flash-exp-image-generation", "short": "2.0-flash-exp-img"},
    {"full": "models/gemini-2.0-flash-lite-001", "short": "2.0-flash-lite-001"},
    {"full": "models/gemini-2.0-flash-lite", "short": "2.0-flash-lite"},
    {"full": "models/gemini-2.0-flash-lite-preview-02-05", "short": "2.0-flash-lite-p-0205"},
    {"full": "models/gemini-2.0-flash-lite-preview", "short": "2.0-flash-lite-p"},
    {"full": "models/gemini-2.0-pro-exp", "short": "2.0-pro-exp"},
    {"full": "models/gemini-2.0-pro-exp-02-05", "short": "2.0-pro-exp-0205"},
    {"full": "models/gemini-exp-1206", "short": "exp-1206"},
    {"full": "models/gemini-2.0-flash-thinking-exp-01-21", "short": "2.0-flash-think-0121"},
    {"full": "models/gemini-2.0-flash-thinking-exp", "short": "2.0-flash-think-exp"},
    {"full": "models/gemini-2.0-flash-thinking-exp-1219", "short": "2.0-flash-think-1219"}
]



# --- Utility Functions ---

def clean_text(text: str) -> str:
    """Cleans the text by removing extra whitespace and non-alphanumeric characters."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove non-alphanumeric, keep basic punctuation
    return text.strip()

def split_text(text: str, chunk_size: int = config.chunk_size) -> List[str]:
    """Splits the text into chunks of a specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def ocr_image(image_path: str) -> str:
    """Performs OCR on an image using the fixed OCR model."""
    try:
        ocr_model = genai.GenerativeModel(config.fixed_ocr_model)
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
        response = ocr_model.generate_content(
            ["Extract text from this image:", {"mime_type": "image/jpeg", "data": img_data}]
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return f"❌ OCR Error: {e}"  # Return error message as string

def generate_content(text_chunk: str) -> str:
    """Generates content (questions, answers, etc.) from a text chunk using Gemini."""
    if not config.model:
        return "❌ Gemini model not initialized.  Check API key."

    try:
        prompt = config.get_prompt(text_chunk)
        response = config.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=config.max_tokens)
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        return f"❌ Error: {e}"


def parse_mcq(text: str) -> List[Dict[str, Union[str, List[str], int]]]:
    """Parses MCQ output from Gemini into a structured list of dictionaries."""
    mcqs = []
    pattern = r"Q: (.+?)\nA: (.+?)\nB: (.+?)\nC: (.+?)\nD: (.+?)\nCorrect: ([ABCD])\nE: (.+?)(?=\nQ:|$)"
    matches = re.finditer(pattern, text, re.DOTALL)
    for match in matches:
        question = match.group(1).strip()[:256]
        options = [match.group(i).strip()[:100] for i in range(2, 6)]
        correct_option_index = {"A": 0, "B": 1, "C": 2, "D": 3}[match.group(6).strip()]
        explanation = match.group(7).strip()[:1024]
        mcqs.append({
            "question": question,
            "options": options,
            "correct": correct_option_index,
            "explanation": explanation,
        })
    return mcqs

def parse_qa(text: str) -> List[Dict[str, str]]:
    """Parses simple Q&A output."""
    qas = []
    pattern = r"\*\*Q:\*\* (.+?) \| \*\*A:\*\* (.+?) \| \*\*E:\*\* (.+?)(?=\n\*\*Q:\*\*|$)"
    matches = re.finditer(pattern, text, re.DOTALL)
    for match in matches:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        explanation = match.group(3).strip()
        qas.append({"question": question, "answer": answer, "explanation": explanation})
    return qas



# --- Telegram Bot Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /start command."""
    await update.message.reply_text(
        "👋 **Hi!** Send me a .txt, .pdf, or image (.jpg/.png) file, and I'll create a quiz from it.\n\n"
        "📌 **How it works:**\n"
        "- Default: MCQ mode\n"
        "- **Simple Mode**: Set a manual prompt for simple Q&A.\n"
        "- **MCQ Mode**: Exam-oriented MCQs covering the entire topic.\n"
        "- Change modes with `/setmode simple` or `/setmode mcq`.\n\n"
        "🔹 **Commands:**\n"
        "- `/start` - Start the bot\n"
        "- `/setkey <key>` - Set your Gemini API key\n"
        "- `/listmodels` - See and select available models\n"
        "- `/setmode <simple/mcq/custom>` - Set quiz mode\n"
        "- `/setprompt <prompt>` - Set a custom prompt for the 'custom' mode.\n"
        "- `/help` - See all commands",
        parse_mode="Markdown",
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /help command."""
    await update.message.reply_text(
        "📌 **All Commands:**\n\n"
        "- `/start` - Start the bot\n"
        "- `/setkey <key>` - Set your Gemini API key\n"
        "- `/listmodels` - See and select available Gemini models\n"
        "- `/setmode <simple/mcq/custom>` - Set quiz mode\n"
        "- `/setprompt <prompt>` - Set a custom prompt (for 'custom' mode)\n"
        "- `/help` - Show this message",
        parse_mode="Markdown",
    )

async def set_key(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /setkey command."""
    if not context.args:
        await update.message.reply_text(
            "❌ Please provide your API key. Example: `/setkey your-api-key`", parse_mode="Markdown"
        )
        return
    api_key = " ".join(context.args)
    config.set_api_key(api_key)
    await update.message.reply_text(
        "✅ Gemini API key set. Now choose a model: `/listmodels` or use the default.", parse_mode="Markdown"
    )

async def list_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /listmodels command."""
    keyboard = [
        [InlineKeyboardButton(model["short"], callback_data=f"model_{model['full']}")]
        for model in AVAILABLE_MODELS
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "📌 **Available Gemini Models:**\nPlease select a model:",
        reply_markup=reply_markup,
        parse_mode="Markdown",
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles button presses (for model selection and processing options)."""
    query = update.callback_query
    await query.answer()
    # Removed:  query.message.reply_to_message.text  # This caused the error.

    if query.data.startswith("model_"):
        model_name = query.data[len("model_") :]
        config.set_model(model_name)
        await query.edit_message_text(
            f"✅ Model `{model_name}` set. Now send a file or image.", parse_mode="Markdown"
        )
        return # Added return to prevent further processing

    # Use context.user_data to get the stored text content
    text_content = context.user_data.get("text_content")
    if not text_content:
        await query.edit_message_text("❌ No text content found. Please send the file/image again.", parse_mode="Markdown")
        return

    if query.data == "ocr":
        # OCR and send as text file
        if text_content:
            with open("ocr_output.txt", "w", encoding="utf-8") as f:
                f.write(text_content)
            await context.bot.send_document(chat_id=query.message.chat_id, document=open("ocr_output.txt", "rb"))
            os.remove("ocr_output.txt")  # Clean up
            await query.edit_message_text("✅ OCR text sent.", parse_mode="Markdown") # Indicate completion

        else:
            await query.edit_message_text("❌ No OCR text available.", parse_mode="Markdown")
        return

    elif query.data in ["mcq", "qa", "custom"]:
        # Set mode and process
        config.set_mode(query.data)
        await process_text_and_generate_output(update, context, text_content)  # Pass context and text_content
        # await query.message.delete() #Removed to show generated content
        return # Added return to prevent further processing


async def set_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /setmode command."""
    if not context.args or context.args[0].lower() not in ["simple", "mcq", "custom"]:
        await update.message.reply_text(
            "❌ Please choose `simple`, `mcq` or 'custom'. Example: `/setmode mcq`", parse_mode="Markdown"
        )
        return
    mode = context.args[0].lower()

    if mode == "simple":
        config.set_mode("qa")  # 'simple' is an alias for 'qa'.
    else:
        config.set_mode(mode)  # 'mcq' or 'custom'

    await update.message.reply_text(f"✅ Mode `{mode}` set. Now send a file or image.", parse_mode="Markdown")


async def set_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles /setprompt"""
    if not context.args:
        await update.message.reply_text("❌ Please provide a prompt. Example: `/setprompt My custom prompt`", parse_mode="Markdown")
        return
    prompt_text = " ".join(context.args)
    config.set_custom_prompt(prompt_text)  # Sets mode to custom automatically
    await update.message.reply_text("✅ Custom prompt set.  Now send a file or image.", parse_mode="Markdown")

# Replace the existing `process_text_and_generate_output` function with this updated version:

async def process_text_and_generate_output(update: Update, context: ContextTypes.DEFAULT_TYPE, text_content: str):
    """Processes the extracted text and generates the quiz or output."""
    # Use query.message for consistency, even if it's from a callback
    if update.callback_query:
        message = update.callback_query.message
    else:
        message = update.message

    progress_msg = await message.reply_text("🔄 **Processing...**", parse_mode="Markdown")

    try:
        text_content = clean_text(text_content)
        if not text_content:
            await progress_msg.edit_text("❌ No text could be extracted.", parse_mode="Markdown")
            return
        chunks = split_text(text_content)
        total_chunks = len(chunks)
        output_text = ""

        for i, chunk in enumerate(chunks):
            await progress_msg.edit_text(f"🔄 **Progress:** Processing chunk {i + 1}/{total_chunks}...", parse_mode="Markdown")
            output_text += generate_content(chunk) + "\n\n"

            # Cooldown logic: Apply 30-40 seconds wait only for 'pro' models, skip for 'flash' models
            if config.is_pro_model() and i < total_chunks - 1:  # Only wait if not the last chunk
                wait_time = random.uniform(30, 40)  # Random wait between 30 and 40 seconds
                await progress_msg.edit_message_text(
                    f"⏳ **Waiting:** Chunk {i + 1}/{total_chunks} complete. Waiting {wait_time:.1f} seconds due to rate limits...",
                    parse_mode="Markdown"
                )
                time.sleep(wait_time)

        if config.mode == "mcq":
            mcqs = parse_mcq(output_text)
            if not mcqs:
                await progress_msg.edit_text("❌ Error parsing MCQs. Check the prompt.", parse_mode="Markdown")
                return

            await progress_msg.edit_text(f"✅ **MCQ Quiz Ready!** ({len(mcqs)} questions)", parse_mode="Markdown")
            for i, mcq in enumerate(mcqs, 1):
                logger.info(
                    f"Poll {i}: Q: {mcq['question']} | Options: {mcq['options']} | Correct: {mcq['correct']} | E: {mcq['explanation']}"
                )
                logger.info(
                    f"Lengths - Q: {len(mcq['question'])}, Options: {[len(opt) for opt in mcq['options']]}, E: {len(mcq['explanation'])}"
                )

                question = mcq['question'][:256]
                options = [opt[:100] for opt in mcq['options']][:10]
                explanation = mcq['explanation'][:1024]

                if len(question) > 256 or any(len(opt) > 100 for opt in options) or len(explanation) > 1024:
                    logger.error(f"Invalid lengths in question {i}")
                    continue

                try:
                    await message.reply_poll(
                        question=f"Question {i}: {question}",
                        options=options,
                        type=Poll.QUIZ,
                        correct_option_id=mcq["correct"],
                        is_anonymous=False,
                        explanation=explanation,
                        explanation_parse_mode="Markdown",
                    )
                except Exception as e:
                    logger.error(f"Poll {i} error: {e}")
                    await send_text_fallback(message, mcq, i, e)

        elif config.mode == "qa":
            qas = parse_qa(output_text)
            if not qas:
                await progress_msg.edit_text("❌ Error parsing Q&A. Check prompt.", parse_mode="Markdown")
                return
            await send_formatted_qa(message, qas, progress_msg)

        elif config.mode == "custom":
            await send_formatted_custom(message, output_text, progress_msg)

    except Exception as e:
        logger.error(f"Quiz generation error: {e}")
        await progress_msg.edit_text(f"❌ Processing error: {e}", parse_mode="Markdown")

async def send_text_fallback(message: Update, mcq: Dict, question_number: int, error: Exception):
    """Sends a text-based fallback if a poll fails."""
    text_output = (
        f"**Question {question_number}:** {mcq['question']}\n"
        f"**A:** {mcq['options'][0]}\n"
        f"**B:** {mcq['options'][1]}\n"
        f"**C:** {mcq['options'][2]}\n"
        f"**D:** {mcq['options'][3]}\n"
        f"**Correct Answer:** {mcq['options'][mcq['correct']]}\n"
        f"**Explanation:** {mcq['explanation']}\n"
    )
    # Send in chunks if too long
    if len(text_output) > 4000:
        parts = [text_output[i:i + 4000] for i in range(0, len(text_output), 4000)]
        for part in parts:
            await message.reply_text(part, parse_mode="Markdown")
    else:
        await message.reply_text(text_output, parse_mode="Markdown")

    await message.reply_text(
        f"⚠️ Question {question_number} was sent as text instead of a poll due to an error: {error}",
        parse_mode="Markdown",
    )

async def send_formatted_qa(message: Update, qas: List[Dict], progress_msg):
    """Sends formatted Q&A output, handling long messages."""
    output_parts = []
    current_part = f"✅ **Simple Quiz Ready!** ({len(qas)} questions)\n\n"

    for i, qa in enumerate(qas, 1):
        question_block = (
            f"**Question {i}:** {qa['question']}\n"
            f"**Answer:** {qa['answer']}\n"
            f"**Explanation:** {qa['explanation']}\n\n"
        )
        if len(current_part) + len(question_block) > 4000:
            output_parts.append(current_part)
            current_part = question_block
        else:
            current_part += question_block

    if current_part:  # Append the last part
        output_parts.append(current_part)

    for part in output_parts:
        await message.reply_text(part, parse_mode="Markdown")
    await progress_msg.delete()  # Clean up the progress message

async def send_formatted_custom(message, output_text: str, progress_msg):
    """Sends custom prompt output, using simple formatting."""

    # Split into chunks to handle long output
    output_parts = []
    current_part = ""

    for line in output_text.splitlines():
        if len(current_part) + len(line) + 1 > 4000:  # +1 for the newline
            output_parts.append(current_part)
            current_part = line + "\n"
        else:
            current_part += line + "\n"

    if current_part:
        output_parts.append(current_part)
    await progress_msg.edit_text("✅ **Custom Output Ready!**", parse_mode="Markdown")

    for part in output_parts:
        # Basic formatting:  bold for headings, regular text otherwise.
        formatted_part = ""
        for line in part.splitlines():
            if line.strip().endswith(":"):  # Simple heuristic for headings.
                formatted_part += f"**{line.strip()}**\n"
            else:
                formatted_part += f"{line.strip()}\n"

        await message.reply_text(formatted_part, parse_mode="Markdown")
    await progress_msg.delete()

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles file uploads (text, PDF, images)."""

    progress_msg = await update.message.reply_text("⏳ **Downloading file...**", parse_mode="Markdown")
    try:
        if update.message.document:
            file = update.message.document
            file_obj = await context.bot.get_file(file.file_id)
            file_path = await file_obj.download_to_drive()

            text_content = ""  # Store extracted text here.

            if file.mime_type == "text/plain":
                await progress_msg.edit_text("📝 **Reading .txt file...**", parse_mode="Markdown")
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            elif file.mime_type == "application/pdf":
                await progress_msg.edit_text("📄 **Processing .pdf file...**", parse_mode="Markdown")
                images = convert_from_path(file_path)
                total_pages = len(images)
                for i, image in enumerate(images):
                    image_path = f"temp_page_{i}.jpg"
                    image.save(image_path, "JPEG")
                    await progress_msg.edit_text(
                        f"🖼️ **OCR processing page {i + 1}/{total_pages}...**", parse_mode="Markdown"
                    )
                    text_content += ocr_image(image_path) + "\n"
                    os.remove(image_path)  # Clean up temp image
            else:
                await progress_msg.edit_text("❌ Unsupported file type. Please send .txt, .pdf, or .jpg/.png files.", parse_mode="Markdown")
                os.remove(file_path)
                return

            os.remove(file_path)  # Clean up downloaded file


        # Create the option buttons *after* OCR
        keyboard = [
            [InlineKeyboardButton("Get OCR Text", callback_data="ocr")],
            [InlineKeyboardButton("Generate MCQs", callback_data="mcq")],
            [InlineKeyboardButton("Generate Q&A", callback_data="qa")],
            [InlineKeyboardButton("Custom Prompt", callback_data="custom")]

        ]

        reply_markup = InlineKeyboardMarkup(keyboard)
        # Store text_content in context.user_data
        context.user_data["text_content"] = text_content

        await update.message.reply_text(
            "✅ File processed. Choose an action:", reply_markup=reply_markup, parse_mode="Markdown"
        )
        await progress_msg.delete()  # Clean up progress message


    except Exception as e:
        logger.error(f"File processing error: {e}")
        await progress_msg.edit_text(f"❌ Processing error: {e}", parse_mode="Markdown")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles photo uploads (performs OCR)."""
    progress_msg = await update.message.reply_text("⏳ **Downloading photo...**", parse_mode="Markdown")
    try:
        photo = update.message.photo[-1]  # Get the largest image
        file_obj = await photo.get_file()
        file_path = await file_obj.download_to_drive()

        await progress_msg.edit_text("🖼️ **Performing OCR on photo...**", parse_mode="Markdown")
        text_content = ocr_image(file_path)  # OCR the image
        os.remove(file_path)
        # Create the option buttons *after* OCR
        keyboard = [
            [InlineKeyboardButton("Get OCR Text", callback_data="ocr")],
            [InlineKeyboardButton("Generate MCQs", callback_data="mcq")],
            [InlineKeyboardButton("Generate Q&A", callback_data="qa")],
            [InlineKeyboardButton("Custom Prompt", callback_data="custom")]

        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Store text_content in context.user_data
        context.user_data["text_content"] = text_content
        await update.message.reply_text(
            "✅ Photo processed. Choose an action:",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
        await progress_msg.delete()

    except Exception as e:
        logger.error(f"Photo processing error: {e}")
        await progress_msg.edit_text(f"❌ Processing error: {e}", parse_mode="Markdown")
        if os.path.exists(file_path):  # Clean up if download succeeded, but processing failed.
            os.remove(file_path)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Error handler."""
    logger.error(f"Bot error: {context.error}")
    if update and update.message:
        await update.message.reply_text(f"❌ An error occurred: {context.error}", parse_mode="Markdown")

# --- Main Function ---

def main():
    """Main function to start the bot."""
    app = Application.builder().token("7570080066:AAGzxX9TP0V0zEwZNuFRVih8XV9goXnCssA").build()

    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("setkey", set_key))
    app.add_handler(CommandHandler("listmodels", list_models))
    app.add_handler(CommandHandler("setmode", set_mode))
    app.add_handler(CommandHandler("setprompt", set_prompt))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(CallbackQueryHandler(button_handler))  # Single handler for *all* buttons.
    app.add_error_handler(error)
    app.run_polling()

if __name__ == "__main__":
    main()