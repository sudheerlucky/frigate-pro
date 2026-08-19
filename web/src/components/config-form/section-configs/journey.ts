import type { SectionConfigOverrides } from "./types";

const journey: SectionConfigOverrides = {
  base: {
    fieldOrder: [
      "show_in_nav",
      "min_appearance_confidence",
      "ambiguity_margin",
    ],
    hiddenFields: [],
    advancedFields: ["ambiguity_margin"],
    overrideFields: [],
    restartRequired: [],
    fieldMessages: [],
    uiSchema: {
      min_appearance_confidence: {
        "ui:options": { size: "sm" },
      },
      ambiguity_margin: {
        "ui:options": { size: "sm" },
      },
    },
  },
};

export default journey;
