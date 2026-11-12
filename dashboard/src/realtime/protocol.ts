import { z } from "zod";

const envelope = z.object({
  type: z.enum(["STATE", "SNAPSHOT", "METRICS", "EVENT", "HEARTBEAT"]),
  timestamp: z.number().optional(),
  data: z.unknown().optional(),
});

export type SocketMessage = z.infer<typeof envelope>;

export function parseSocketMessage(raw: string): SocketMessage | null {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  const parsed = envelope.safeParse(value);
  return parsed.success ? parsed.data : null;
}
