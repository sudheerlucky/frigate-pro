import { useMemo, useState } from "react";
import useSWR from "swr";
import { useTranslation } from "react-i18next";
import { baseUrl } from "@/api/baseUrl";
import { formatUnixTimestampToDateTime } from "@/utils/dateUtil";
import { Event } from "@/types/event";
import JourneyView from "@/components/overlay/JourneyView";

export default function Journeys() {
  const { t } = useTranslation(["views/explore"]);

  const { data: events, isLoading } = useSWR<Event[]>([
    "events",
    { labels: "person", limit: 100 },
  ]);

  const feed = useMemo(
    () => (events ?? []).slice().sort((a, b) => b.start_time - a.start_time),
    [events],
  );

  const [selected, setSelected] = useState<Event | null>(null);
  const [open, setOpen] = useState(false);

  return (
    <div className="scrollbar-container flex size-full flex-col gap-3 overflow-y-auto p-2 md:p-4">
      <div className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold text-primary">
          {t("journeysPage.title")}
        </h1>
        <p className="max-w-xl text-sm text-secondary-foreground">
          {t("journeysPage.desc")}
        </p>
      </div>

      {isLoading && (
        <div className="text-sm text-secondary-foreground">
          {t("journeysPage.loading")}
        </div>
      )}

      {!isLoading && feed.length === 0 && (
        <div className="text-sm text-secondary-foreground">
          {t("journeysPage.empty")}
        </div>
      )}

      <div className="flex flex-col gap-2">
        {feed.map((event) => (
          <button
            key={event.id}
            className="flex items-center gap-3 rounded-lg bg-secondary p-2 text-left transition-colors hover:bg-muted"
            onClick={() => {
              setSelected(event);
              setOpen(true);
            }}
          >
            <img
              className="h-14 w-14 shrink-0 rounded-md object-cover"
              src={`${baseUrl}api/events/${event.id}/thumbnail.webp`}
              alt={event.camera}
            />
            <div className="flex min-w-0 flex-col gap-0.5">
              <div className="truncate text-sm font-medium text-primary">
                {event.sub_label && event.sub_label !== "unknown"
                  ? event.sub_label
                  : t("journeysPage.unknownPerson")}
              </div>
              <div className="truncate text-xs text-secondary-foreground">
                {event.camera}
                {" · "}
                {formatUnixTimestampToDateTime(event.start_time, {
                  time_style: "medium",
                  date_style: "medium",
                })}
              </div>
            </div>
          </button>
        ))}
      </div>

      <JourneyView
        eventId={open ? (selected?.id ?? null) : null}
        sourceCamera={selected?.camera}
        open={open}
        setOpen={setOpen}
      />
    </div>
  );
}
