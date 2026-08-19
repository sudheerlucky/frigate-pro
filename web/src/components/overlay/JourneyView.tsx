import { useCallback, useEffect, useState } from "react";
import axios from "axios";
import { useTranslation } from "react-i18next";
import { baseUrl } from "@/api/baseUrl";
import { cn } from "@/lib/utils";
import { formatUnixTimestampToDateTime } from "@/utils/dateUtil";
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
  start_time: number;
  hop: number;
};

type JourneyResponse = {
  success: boolean;
  event_id: string;
  start_time: number;
  matches: JourneyMatch[];
};

type JourneyTimelineItem = {
  eventId: string;
  camera?: string;
  startTime: number;
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
  const [timeline, setTimeline] = useState<JourneyTimelineItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !eventId) {
      return;
    }

    setSelectedEventId(eventId);
    setTimeline([]);
    setLoading(true);

    axios
      .get<JourneyResponse>(`events/${eventId}/likely_journey`)
      .then((resp) => {
        const source: JourneyTimelineItem = {
          eventId: resp.data.event_id,
          camera: sourceCamera,
          startTime: resp.data.start_time,
          isSource: true,
        };
        const matched: JourneyTimelineItem[] = (resp.data.matches ?? []).map(
          (m) => ({
            eventId: m.event_id,
            camera: m.camera,
            startTime: m.start_time,
            match: m,
            isSource: false,
          }),
        );
        setTimeline(
          [source, ...matched].sort((a, b) => a.startTime - b.startTime),
        );
      })
      .finally(() => setLoading(false));
  }, [open, eventId, sourceCamera]);

  const selectEvent = useCallback((id: string) => {
    setSelectedEventId(id);
  }, []);

  if (!eventId) {
    return null;
  }

  const matchCount = timeline.filter((item) => !item.isSource).length;

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="scrollbar-container max-h-[90dvh] max-w-3xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{t("journey.title")}</DialogTitle>
          <DialogDescription>{t("journey.desc")}</DialogDescription>
        </DialogHeader>

        <div className="relative flex gap-0 overflow-x-auto pb-2 pt-1">
          {timeline.map((item, index) => (
            <div key={item.eventId} className="flex shrink-0 items-center">
              {index > 0 && (
                <div className="flex shrink-0 flex-col items-center gap-0.5 self-center">
                  {item.match?.hop !== undefined && (
                    <div className="whitespace-nowrap font-mono text-[9px] text-secondary-foreground">
                      {t("journey.hop", { number: item.match.hop })}
                    </div>
                  )}
                  <div className="h-px w-6 bg-secondary sm:w-10" />
                </div>
              )}
              <button
                className={cn(
                  "flex shrink-0 flex-col items-center gap-1 rounded-lg border-2 p-1",
                  selectedEventId === item.eventId
                    ? "border-selected"
                    : "border-transparent",
                )}
                onClick={() => selectEvent(item.eventId)}
              >
                <div className="font-mono text-[10px] text-secondary-foreground">
                  {formatUnixTimestampToDateTime(item.startTime, {
                    time_style: "medium",
                  })}
                </div>
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
            </div>
          ))}
        </div>

        {loading && (
          <div className="text-sm text-secondary-foreground">
            {t("journey.loading")}
          </div>
        )}

        {!loading && matchCount === 0 && (
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
            playsInline
            src={`${baseUrl}api/events/${selectedEventId}/clip.mp4`}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}
