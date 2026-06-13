/**
 * Server-side helper: proxies a fetch to the FastAPI backend.
 * Used exclusively in Route Handlers (never imported by client components).
 */

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function backendFetch(
  path: string,
  init?: RequestInit
): Promise<Response> {
  return fetch(`${BACKEND}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
}
