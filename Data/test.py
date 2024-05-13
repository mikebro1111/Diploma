import asyncio
import time
import logging

from aiogram import Bot, Dispatcher, types, F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import CommandStart, Command
from aiogram.utils.keyboard import ReplyKeyboardBuilder

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOKEN = "" #Token

bot = Bot(token=TOKEN)
dp = Dispatcher(bot=bot)


@dp.message(CommandStart())
async def start_handler(message: types.Message):
    user_id = message.from_user.id
    user_full_name = message.from_user.full_name
    logging.info(f'{user_id} {user_full_name}', time.asctime())
    await message.reply(
        f"Hello, {user_full_name}! I am your investment assistant. Below are buttons where you can enter your data and I will help you choose a company for your investment!")


@dp.message(Command(commands="test_click_btn"))
async def start(message: types.Message):
    builder = ReplyKeyboardBuilder()
    builder.button(text='button_clicked')

    await message.answer("Click the button:", reply_markup=builder.as_markup())


@dp.message(Command(commands="test_click_inline"))
async def start(message: types.Message):
    button = types.InlineKeyboardButton(text="Beginner", callback_data="Beginner")
    button2 = types.InlineKeyboardButton(text="Intermediate", callback_data="Intermediate")
    button3 = types.InlineKeyboardButton(text="Experienced", callback_data="Experienced")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[button, button2, button3]])

    await message.answer("Type of investor", reply_markup=keyboard)

@dp.message(Command(commands="test_click_inline"))
async def start(message: types.Message):
    button = types.InlineKeyboardButton(text="Saving money", callback_data="Saving money")
    button2 = types.InlineKeyboardButton(text="Capital increase", callback_data="Capital increase")
    button3 = types.InlineKeyboardButton(text="Income for retirement", callback_data="Income for retirement")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[button, button2, button3]])

    await message.answer("Financial goals", reply_markup=keyboard)

@dp.message(Command(commands="test_click_inline"))
async def start(message: types.Message):
    button = types.InlineKeyboardButton(text="Conservative(0-30%)", callback_data="Conservative(0-30%)")
    button2 = types.InlineKeyboardButton(text="Moderate(30-60%)", callback_data="Moderate(30-60%)")
    button3 = types.InlineKeyboardButton(text="Aggressive(60+%)", callback_data="Aggressive(60+%)")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[button, button2, button3]])

    await message.answer("Percentage of risk", reply_markup=keyboard)

@dp.message(Command(commands="test_click_inline"))
async def start(message: types.Message):
    button = types.InlineKeyboardButton(text="Short term(<6 months-2 years)", callback_data="Short term(<6 months-2 years)")
    button2 = types.InlineKeyboardButton(text="Average term(2-4 years)", callback_data="Average term(2-4 years)")
    button3 = types.InlineKeyboardButton(text="Long term(4-5+ years)", callback_data="Long term(4-5+ years)")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[button, button2, button3]])

    await message.answer("Investment term", reply_markup=keyboard)

@dp.message(Command(commands="test_click_inline"))
async def start(message: types.Message):
    button = types.InlineKeyboardButton(text="Asia", callback_data="Asia")
    button2 = types.InlineKeyboardButton(text="Europe", callback_data="Europe")
    button3 = types.InlineKeyboardButton(text="America", callback_data="America")
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[button, button2, button3]])

    await message.answer("Region selection", reply_markup=keyboard)

@dp.callback_query()
async def callback_query_inline(call: types.CallbackQuery):
    await call.message.answer(f"Your choice: {call.data}")

@dp.message(lambda message: message.text == 'button_clicked')
async def process_callback_button(callback_query: types.CallbackQuery):
    print(callback_query)


async def main():
    await dp.start_polling(bot)

dataset = pd.read_csv('data/data/data.csv') # -> ...data/data/data.cvs)

vectorizer = TfidfVectorizer()
print(dataset.columns)
X = vectorizer.fit_transform(dataset['text'])

def choose_response(input_text):
    input_vector = vectorizer.transform([input_text])
    similarities = cosine_similarity(input_vector, X)
    best_match_index = similarities.argmax()
    return dataset.iloc[best_match_index]['response']

async def handle_message(message: types.Message):
    input_text = message.text
    response = choose_response(input_text)
    await message.answer(response)

@dp.message_handler()
async def process_message(message: types.Message):
    await handle_message(message)

if __name__ == '__main__':
    asyncio.run(main())
