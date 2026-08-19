import type { SectionConfigOverrides } from "./types";

const journey: SectionConfigOverrides = {
  base: {
    fieldOrder: ["min_appearance_confidence"],
    hiddenFields: [],
    advancedFields: [],
    overrideFields: [],
    restartRequired: [],
    fieldMessages: [],
    uiSchema: {
      min_appearance_confidence: {
        "ui:options": { size: "sm" },
      },
    },
  },
};

export default journey;
