import type { SectionConfigOverrides } from "./types";

const tapo: SectionConfigOverrides = {
  base: {
    fieldOrder: ["enabled", "host", "username", "password"],
    hiddenFields: [],
    advancedFields: [],
    overrideFields: [],
    restartRequired: [],
    fieldMessages: [],
    uiSchema: {
      host: {
        "ui:options": { size: "sm" },
      },
      username: {
        "ui:options": { size: "sm" },
      },
      password: {
        "ui:widget": "password",
      },
    },
  },
};

export default tapo;
