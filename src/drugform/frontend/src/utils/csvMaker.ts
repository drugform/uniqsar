export const csvMaker = (data: any) => {
  const values = Object.values(data);
  return values.join(",");
};
