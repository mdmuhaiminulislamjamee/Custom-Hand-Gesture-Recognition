'use client';

export const labLabel = (value: string) => value.replaceAll('_', ' ').replace(/\b\w/g, letter => letter.toUpperCase());

export async function labRequest<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const text = await response.text();
  let payload: unknown;
  try { payload = text ? JSON.parse(text) : {}; }
  catch { throw new Error(response.ok ? 'The server returned an invalid response.' : `The request failed (${response.status}). Check that the local backend is running.`); }
  if (!response.ok) {
    const detail = payload && typeof payload === 'object' && 'detail' in payload ? payload.detail : null;
    const message = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(item => typeof item === 'object' && item && 'msg' in item ? String(item.msg) : String(item)).join('; ') : `The request failed (${response.status}).`;
    throw new Error(message);
  }
  return payload as T;
}

const readable = (value: unknown) => {
  if (value === null || value === undefined) return 'Not available';
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : value.toLocaleString(undefined, { maximumFractionDigits: 5 });
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  return String(value);
};

function valuesIn(value: unknown, path = '', depth = 0): [string, unknown][] {
  if (!value || typeof value !== 'object' || Array.isArray(value) || depth > 4) return [];
  return Object.entries(value).flatMap(([key, child]): [string, unknown][] => {
    const name = path ? `${path} / ${labLabel(key)}` : labLabel(key);
    if (Array.isArray(child)) return [];
    if (child && typeof child === 'object') return valuesIn(child, name, depth + 1);
    if (typeof child === 'string' && child.length > 180) return [];
    return [[name, child]];
  });
}

export function LabReport({ result, title = 'Results' }: { result: unknown; title?: string }) {
  const comparisons = result && typeof result === 'object' ? Object.entries(result).filter(([key, value]) => ['validation', 'camera_validation', 'public_test', 'camera_test'].includes(key) && value && typeof value === 'object') : [];
  const rows = valuesIn(result).slice(0, 90);
  return <section className="lab-report" aria-label={title}>
    <h3>{title}</h3>
    {comparisons.length > 0 && <div className="table-wrap"><table className="lab-results-table"><thead><tr><th>Dataset</th><th>Model</th><th>Rows</th><th>Macro F1</th><th>Accepted precision</th><th>Unknown false accept rate</th></tr></thead><tbody>{comparisons.flatMap(([dataset, value]) => Object.entries(value as Record<string, Record<string, unknown>>).filter(([model]) => model === 'production' || model === 'candidate').map(([model, metrics]) => <tr key={`${dataset}-${model}`}><td>{labLabel(dataset)}</td><td>{labLabel(model)}</td><td>{readable(metrics.rows ?? metrics.samples ?? metrics.count)}</td><td>{readable(metrics.macro_f1)}</td><td>{readable(metrics.accepted_precision)}</td><td>{readable(metrics.unknown_false_accept_rate)}</td></tr>))}</tbody></table></div>}
    {rows.length > 0 && <div className="table-wrap"><table className="lab-results-table"><thead><tr><th>Measure</th><th>Value</th></tr></thead><tbody>{rows.map(([key, value]) => <tr key={key}><td>{key}</td><td>{readable(value)}</td></tr>)}</tbody></table></div>}
    <details><summary>Full report and per-class details</summary><pre>{JSON.stringify(result, null, 2)}</pre></details>
    <a className="lab-text-link" download="gesture-lab-report.json" href={`data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(result, null, 2))}`}>Download report JSON</a>
  </section>;
}
