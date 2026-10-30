/** Auxiliary expression cadence; independent of hand/body and Holistic detection. */
export function parseFaceEvery(value: string | null): number {
  const n = Number(value);
  return Number.isSafeInteger(n) && n >= 1 ? n : 3;
}
