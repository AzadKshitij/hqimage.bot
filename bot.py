import os
# from xml.dom.minidom import Document
from telefone import Bot, Message
from telefone.api.utils import File
from telefone.tools.text import html, ParseMode
import logging
# import requests
from dotenv import load_dotenv

# For the model
import torch
import numpy as np
import PIL.Image as pil_image

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

token = os.getenv('TOKEN')
get_file_env = os.getenv('GET_FILE')
file_path_env = os.getenv('FILE_PATH')
save_path_env = os.getenv('SAVE_PATH')


model_file = os.getenv('MODEL_FILE')
scale = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(device)
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()

bot = Bot(token)
bot.labeler.vbml_ignore_case = True


@bot.on.message()
async def handler(msg: Message) -> str:
    # check if the messge contains image file

    if msg.photo:
        try:
            image = msg.photo[-1].file_id

            get_file = await msg.ctx_api.http_client.request_json(
                get_file_env.format(token, image))
            file_path = get_file['result']['file_path']
            image = await msg.ctx_api.http_client.request_content(
                file_path_env.format(token, file_path))

            with open(file_path, "wb") as f:
                f.write(image)

            image = pil_image.open(file_path).convert("RGB")
            image = np.array(image).astype(np.float32)
            ycbcr = convert_rgb_to_ycbcr(image)

            y = ycbcr[..., 0]
            y /= 255.0
            y = torch.from_numpy(y).to(device)
            y = y.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                preds = model(y).clamp(0.0, 1.0)

            preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

            output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]
                              ).transpose([1, 2, 0])
            output = np.clip(convert_ycbcr_to_rgb(output),
                             0.0, 255.0).astype(np.uint8)
            output = pil_image.fromarray(output)
            output.save(save_path_env.format(file_path))
            await msg.ctx_api.send_photo(
                chat_id=msg.chat.id,
                caption="Your image has been processed",
                photo=await File.from_path(save_path_env.format(file_path)),
            )

            os.remove(file_path)
            os.remove(save_path_env.format(file_path))

        except Exception as e:
            logging.error(e)
            return "Error :( :- " + str(e)
    elif msg.text.lower() == "/hi":
        return "Howdy, partner!"

    elif msg.text.lower() == "/start":

        await msg.ctx_api.send_photo(
            chat_id=msg.chat.id,
            caption="Input image",
            photo=await File.from_path("input.jpg"),
        )
        await msg.ctx_api.send_photo(
            chat_id=msg.chat.id,
            caption="Your image has been processed",
            photo=await File.from_path("output.jpg"),
        )
        await msg.answer(
            html.bold("These are the commands you can use:")
            + "\n" + "\n" +
            html.bold("/help") + "\n" + " - Show this help message"
            + "\n" + "\n" +
            html.bold("/start") + "\n" + " - Start a conversation"
            + "\n" + "\n" +
            html.bold("/hi") + "\n" + " - Say hi to the bot"
            + "\n" + "\n" +
            html.italic("Send image if you want to improve the quality ")
            + "\n" + "\n" +
            html.italic(
                """Make sure that the image is actually blurry or it will not 
                work the way you want look at the above example"""),

            parse_mode=ParseMode.HTML,
        )

    else:
        await msg.answer(
            html.bold("These are the commands you can use:")
            + "\n" + "\n" +
            html.bold("/help") + "\n" + " - Show this help message"
            + "\n" + "\n" +
            html.bold("/start") + "\n" + " - Start a conversation"
            + "\n" + "\n" +
            html.bold("/hi") + "\n" + " - Say hi to the bot"
            + "\n" + "\n" +
            html.italic("Send image if you want to improve the quality ")
            + "\n" + "\n" +
            html.italic("""Make sure that the image is actually
             blurry or it will not work the way you want."""),

            parse_mode=ParseMode.HTML,
        )

# watch_to_reload("./")
bot.run_forever()
