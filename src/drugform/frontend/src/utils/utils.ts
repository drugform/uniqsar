import dayjs from "dayjs";

export const isValidJson = (json: string) => {
  try {
    JSON.parse(json);
    return true;
  } catch (e) {
    return false;
  }
};

export const getDate = (timestamp?: number | null): string => {
  if (timestamp === undefined || timestamp === null) {
    return "";
  }
  return dayjs(timestamp).format("DD.MM.YYYY HH:mm:ss");
};
