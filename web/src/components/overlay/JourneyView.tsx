import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import { useTranslation } from "react-i18next";
import { baseUrl } from "@/api/baseUrl";
import { cn } from "@/lib/utils";
import { formatUnixTimestampToDateTime } from "@/utils/dateUtil";
import { GenericVideoPlayer } from "../player/GenericVideoPlayer";
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
  end_time: number | null;
  hop: number;
  current_activity?: string | null;
};

type JourneyResponse = {
  success: boolean;
  event_id: string;
  start_time: number;
  end_time: number | null;
  current_activity?: string | null;
  matches: JourneyMatch[];
};

type JourneyTimelineItem = {
  eventId: string;
  camera?: string;
  startTime: number;
  endTime: number | null;
  match?: JourneyMatch;
  isSource: boolean;
  activity?: string | null;
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
          endTime: resp.data.end_time,
          isSource: true,
          activity: resp.data.current_activity,
        };
        const matched: JourneyTimelineItem[] = (resp.data.matches ?? []).map(
          (m) => ({
            eventId: m.event_id,
            camera: m.camera,
            startTime: m.start_time,
            endTime: m.end_time,
            match: m,
            isSource: false,
            activity: m.current_activity,
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

  // iOS Safari does not reliably play progressive mp4 -- use the same HLS
  // VOD URL pattern (hls.js under the hood via GenericVideoPlayer) that
  // SearchDetailDialog's VideoTab already uses successfully for recorded
  // clips, instead of a raw <video src=".../clip.mp4"> tag. Note: no "api/"
  // prefix -- /vod/ is handled directly by nginx-vod-module, not proxied
  // through the same path as the rest of the JSON API.
  const selectedItem = timeline.find(
    (item) => item.eventId === selectedEventId,
  );
  const vodSource = useMemo(() => {
    if (!selectedItem || !selectedItem.camera) {
      return null;
    }
    const endTime = selectedItem.endTime ?? Date.now() / 1000;
    return `${baseUrl}vod/${selectedItem.camera}/start/${selectedItem.startTime}/end/${endTime}/index.m3u8`;
  }, [selectedItem]);

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
                {item.activity && (
                  <div className="rounded bg-secondary px-1.5 py-0.5 text-[10px] capitalize text-secondary-foreground">
                    {item.activity}
                  </div>
                )}
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

        {!loading && matchCount > 0 && (
          <div className="flex flex-col gap-1">
            <a
              className="w-fit rounded-md bg-secondary px-3 py-1.5 text-xs text-primary hover:bg-muted"
              href={`${baseUrl}api/events/${eventId}/journey/merged_clip.mp4`}
              download={`journey-${eventId}.mp4`}
            >
              {t("journey.mergeClip")}
            </a>
            <div className="text-xs text-secondary-foreground">
              {t("journey.mergeClipNote")}
            </div>
          </div>
        )}

        {vodSource && (
          <div className="aspect-video w-full overflow-hidden rounded-lg bg-black">
            <GenericVideoPlayer key={selectedEventId} source={vodSource} />
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
