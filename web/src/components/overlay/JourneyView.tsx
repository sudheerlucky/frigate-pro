import { useCallback, useEffect, useState } from "react";
import axios from "axios";
import { useTranslation } from "react-i18next";
import { baseUrl } from "@/api/baseUrl";
import { cn } from "@/lib/utils";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";

type JourneyMatch = {
  event_id: string;
  camera: string;
  confidence: number;
  method: "face" | "appearance";
};

type JourneyResponse = {
  success: boolean;
  event_id: string;
  matches: JourneyMatch[];
};

type JourneyStripItem = {
  eventId: string;
  camera?: string;
  match?: JourneyMatch;
  isSource: boolean;
};

type JourneyViewProps = {
  eventId: string | null;
  sourceCamera?: string;
  open: boolean;
  setOpen: (open: boolean) => void;
};

export default function JourneyView({
  eventId,
  sourceCamera,
  open,
  setOpen,
}: JourneyViewProps) {
  const { t } = useTranslation(["views/explore"]);
  const [matches, setMatches] = useState<JourneyMatch[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !eventId) {
      return;
    }

    setSelectedEventId(eventId);
    setMatches([]);
    setLoading(true);

    axios
      .get<JourneyResponse>(`events/${eventId}/likely_journey`)
      .then((resp) => {
        setMatches(resp.data.matches ?? []);
      })
      .finally(() => setLoading(false));
  }, [open, eventId]);

  const selectEvent = useCallback((id: string) => {
    setSelectedEventId(id);
  }, []);

  if (!eventId) {
    return null;
  }

  const strip: JourneyStripItem[] = [
    { eventId, camera: sourceCamera, isSource: true },
    ...matches.map((m) => ({
      eventId: m.event_id,
      camera: m.camera,
      match: m,
      isSource: false,
    })),
  ];

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="scrollbar-container max-h-[90dvh] max-w-3xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{t("journey.title")}</DialogTitle>
          <DialogDescription>{t("journey.desc")}</DialogDescription>
        </DialogHeader>

        <div className="scrollbar-container flex gap-2 overflow-x-auto pb-2">
          {strip.map((item) => (
            <button
              key={item.eventId}
              className={cn(
                "flex shrink-0 flex-col items-center gap-1 rounded-lg border-2 p-1",
                selectedEventId === item.eventId
                  ? "border-selected"
                  : "border-transparent",
              )}
              onClick={() => selectEvent(item.eventId)}
            >
              <img
                className="h-20 w-20 rounded-md object-cover"
                src={`${baseUrl}api/events/${item.eventId}/thumbnail.webp`}
                alt={item.camera ?? ""}
              />
              <div className="max-w-20 truncate text-xs text-primary">
                {item.camera}
              </div>
              <div className="text-xs text-secondary-foreground">
                {item.isSource
                  ? t("journey.source")
                  : item.match?.method === "face"
                    ? t("journey.confirmed")
                    : t("journey.likelyMatch", {
                        confidence: Math.round(
                          (item.match?.confidence ?? 0) * 100,
                        ),
                      })}
              </div>
            </button>
          ))}
        </div>

        {loading && (
          <div className="text-sm text-secondary-foreground">
            {t("journey.loading")}
          </div>
        )}

        {!loading && matches.length === 0 && (
          <div className="text-sm text-secondary-foreground">
            {t("journey.noMatches")}
          </div>
        )}

        {selectedEventId && (
          <video
            key={selectedEventId}
            className="aspect-video w-full rounded-lg bg-black"
            controls
            autoPlay
            muted
            src={`${baseUrl}api/events/${selectedEventId}/clip.mp4`}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}
