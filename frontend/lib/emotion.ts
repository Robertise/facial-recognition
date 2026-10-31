// Emotion label -> icon, for every place that shows an emotion.
//
// `EmotionDetector.predict()` returns the capitalised class name, so that is what
// the backend sends and stores in `attendance.emotion` ("Happy", "Sad", ...). The
// map is kept in sync with emotion/labels.json.
import { Angry, Annoyed, FaceSlightlyFrowning, Frown, Laugh, Meh, Smile, type LucideIcon } from "lucide-react";

export const EMOTION_ICON: Record<string, LucideIcon> = {
  Angry: Angry,
  Disgust: Annoyed,
  Fear: FaceSlightlyFrowning,
  Happy: Smile,
  Sad: Frown,
  Surprise: Laugh,
  Neutral: Meh,
};

/** Normalised label ("happy" -> "Happy"); null when missing. */
export function emotionLabel(label?: string | null): string | null {
  if (!label) return null;
  return label.charAt(0).toUpperCase() + label.slice(1).toLowerCase();
}

/** Icon for an emotion label; Meh when the label is unknown. */
export function emotionIcon(label?: string | null): LucideIcon {
  const key = emotionLabel(label);
  return (key && EMOTION_ICON[key]) || Meh;
}
