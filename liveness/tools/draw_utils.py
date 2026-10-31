"""
draw_utils.py
=============
PIL-based text rendering for OpenCV frames – supports Unicode & emoji.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Font paths ────────────────────────────────────────────────────────────────
_EMOJI_FONT_PATH = '/System/Library/Fonts/Apple Color Emoji.ttc'
_TEXT_FONT_PATH  = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'

# Cache loaded fonts
_font_cache: dict = {}


def _get_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    if key not in _font_cache:
        try:
            _font_cache[key] = ImageFont.truetype(path, size)
        except Exception:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def _is_emoji(ch: str) -> bool:
    cp = ord(ch)
    return (0x1F300 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or
            0xFE00  <= cp <= 0xFE0F  or 0x1F900 <= cp <= 0x1F9FF)


def put_text_pil(img_bgr: np.ndarray, text: str, xy: tuple,
                 font_size: int = 26, text_color=(255, 255, 255),
                 bg_color=(0, 0, 0), bg_pad: int = 6) -> np.ndarray:
    """
    Draw Unicode/emoji text on OpenCV BGR image using Pillow.
    Non-emoji chars use Arial Unicode; emoji chars use Apple Color Emoji.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert('RGBA')
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    text_font  = _get_font(_TEXT_FONT_PATH,  font_size)
    emoji_font = _get_font(_EMOJI_FONT_PATH, font_size)

    x, y = xy

    # Measure full text for background using text_font as approximation
    if bg_color is not None:
        bbox = draw.textbbox((x, y), text, font=text_font)
        draw.rectangle(
            [bbox[0]-bg_pad, bbox[1]-bg_pad, bbox[2]+bg_pad, bbox[3]+bg_pad],
            fill=(*bg_color, 200)
        )

    # Draw char by char: emoji font for emoji, text font otherwise
    cx = x
    for ch in text:
        if _is_emoji(ch):
            font = emoji_font
        else:
            font = text_font
        draw.text((cx, y), ch, font=font,
                  fill=(*text_color, 255), embedded_color=True)
        cx += draw.textlength(ch, font=font)

    pil_img = Image.alpha_composite(pil_img, overlay).convert('RGB')
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def put_liveness_emotion(img_bgr: np.ndarray,
                         live_label: str, live_conf: float,
                         emo_label: str | None, emo_emoji: str | None,
                         emo_conf: float | None,
                         font_size: int = 26) -> np.ndarray:
    """
    Convenience wrapper: draws liveness + emotion lines on frame.
    """
    # Liveness color
    if live_label == 'Real':
        live_color = (60, 220, 60)
    elif live_label == 'Fake':
        live_color = (220, 60, 60)
    else:
        live_color = (160, 160, 160)

    live_text = f'Liveness: {live_label}  {live_conf*100:.1f}%'
    img_bgr = put_text_pil(img_bgr, live_text, (10, 10),
                           font_size=font_size, text_color=live_color)

    if emo_label and emo_emoji:
        emo_text = f'Emotion : {emo_label} {emo_emoji}  {emo_conf*100:.1f}%'
        img_bgr = put_text_pil(img_bgr, emo_text, (10, 10 + font_size + 14),
                               font_size=font_size, text_color=(255, 200, 0))

    return img_bgr
