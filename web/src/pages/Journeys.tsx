import { useState } from "react";
import axios from "axios";
import useSWR from "swr";
import { useTranslation } from "react-i18next";
import { baseUrl } from "@/api/baseUrl";
import { formatUnixTimestampToDateTime } from "@/utils/dateUtil";
import JourneyView from "@/components/overlay/JourneyView";

type JourneySessionMatch = {
  event_id: string;
  camera: string;
  same_camera?: boolean;
};

type JourneySession = {
  event_id: string;
  camera: string;
  start_time: number;
  end_time: number | null;
  sub_label: string | null;
  matches: JourneySessionMatch[];
};

type JourneysResponse = {
  success: boolean;
  sessions: JourneySession[];
};

export default function Journeys() {
  const { t } = useTranslation(["views/explore"]);

  const { data, isLoading } = useSWR<JourneysResponse>(
    "journeys",
    (url: string) => axios.get<JourneysResponse>(url).then((r) => r.data),
  );

  const sessions = data?.sessions ?? [];

  const [selected, setSelected] = useState<JourneySession | null>(null);
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

      {!isLoading && sessions.length === 0 && (
        <div className="text-sm text-secondary-foreground">
          {t("journeysPage.empty")}
        </div>
      )}

      <div className="flex flex-col gap-2">
        {sessions.map((session) => {
          const cameraCount =
            1 + new Set(session.matches.map((m) => m.camera)).size;

          return (
            <button
              key={session.event_id}
              className="flex items-center gap-3 rounded-lg bg-secondary p-2 text-left transition-colors hover:bg-muted"
              onClick={() => {
                setSelected(session);
                setOpen(true);
              }}
            >
              <img
                className="size-20 shrink-0 rounded-md object-cover"
                src={`${baseUrl}api/events/${session.event_id}/snapshot.jpg?crop=1&quality=95`}
                alt={session.camera}
                loading="lazy"
                onError={(e) => {
                  const img = e.currentTarget;
                  const fallback = `${baseUrl}api/events/${session.event_id}/thumbnail.webp`;
                  if (img.src !== fallback) {
                    img.src = fallback;
                  }
                }}
              />
              <div className="flex min-w-0 flex-col gap-0.5">
                <div className="truncate text-sm font-medium text-primary">
                  {session.sub_label && session.sub_label !== "unknown"
                    ? session.sub_label
                    : t("journeysPage.unknownPerson")}
                </div>
                <div className="truncate text-xs text-secondary-foreground">
                  {session.camera}
                  {" · "}
                  {formatUnixTimestampToDateTime(session.start_time, {
                    time_style: "medium",
                    date_style: "medium",
                  })}
                </div>
                {session.matches.length > 0 && (
                  <div className="truncate text-xs text-secondary-foreground">
                    {cameraCount > 1
                      ? t("journeysPage.crossedCameras", { count: cameraCount })
                      : t("journeysPage.seenMultipleTimes", {
                          count: session.matches.length + 1,
                        })}
                  </div>
                )}
              </div>
            </button>
          );
        })}
      </div>

      <JourneyView
        eventId={open ? (selected?.event_id ?? null) : null}
        sourceCamera={selected?.camera}
        open={open}
        setOpen={setOpen}
      />
    </div>
  );
}
