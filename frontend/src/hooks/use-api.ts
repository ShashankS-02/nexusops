"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { isDemoMode } from "@/lib/demo-engine";

interface UseApiOptions {
  /** Polling interval in ms. Omit or set to 0 to disable polling. */
  pollInterval?: number;
}

interface UseApiResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

/**
 * Generic data-fetching hook with optional polling.
 * Falls back gracefully — loading shows skeletons, error surfaces a message.
 */
export function useApi<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = [],
  options: UseApiOptions = {}
): UseApiResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);

  const run = useCallback(async () => {
    try {
      const result = await fetcher();
      if (mounted.current) {
        setData(result);
        setError(null);
      }
    } catch (e) {
      if (mounted.current) {
        setError(e instanceof Error ? e.message : "Unknown error");
      }
    } finally {
      if (mounted.current) setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    mounted.current = true;
    setLoading(true);
    run();

    // In demo mode the lifecycle advances every few seconds, so poll faster
    // (1.5s) than the component's configured interval to keep the pipeline
    // illumination smooth. Reads are in-memory, so this is cheap.
    if (options.pollInterval && options.pollInterval > 0) {
      const interval = isDemoMode() ? 1500 : options.pollInterval;
      const id = setInterval(run, interval);
      return () => {
        mounted.current = false;
        clearInterval(id);
      };
    }
    return () => {
      mounted.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run]);

  return { data, loading, error, refetch: run };
}
