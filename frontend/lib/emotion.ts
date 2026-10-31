// Emotion label -> emoji, for places that only have the stored label string
// (e.g. the attendance history table) rather than the live websocket payload.
//
// `EmotionDetector.predict()` returns the capitalised class name, so that is what
// the backend stores in `attendance.emotion` ("Happy", "Sad", ...). The map is
// kept in sync with emotion/labels.json.

export const EMOTION_EMOJI: Record<string, string> = {
  Angry: "\u{1F620}",
  Disgust: "\u{1F922}",
  Fear: "\u{1F628}",
  Happy: "\u{1F60A}",
  Sad: "\u{1F622}",
  Surprise: "\u{1F632}",
  Neutral: "\u{1F610}",
};

/** Emoji for an emotion label; "" when the label is missing/unknown. */
export function emotionEmoji(label?: string | null): string {
  if (!label) return "";
  if (EMOTION_EMOJI[label]) return EMOTION_EMOJI[label];
  const normalised = label.charAt(0).toUpperCase() + label.slice(1).toLowerCase();
  return EMOTION_EMOJI[normalised] ?? "";
}
