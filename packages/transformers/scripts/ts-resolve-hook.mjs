import { existsSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const TS_EXTENSIONS = [".ts", ".tsx", ".mts"];

function isRelativeSpecifier(specifier) {
  return specifier.startsWith("./") || specifier.startsWith("../");
}

function hasExtension(specifier) {
  const noQuery = specifier.split("?")[0];
  return path.extname(noQuery) !== "";
}

/**
 * Resolve extensionless relative imports to .ts / .tsx / .mts so `node` can run
 * package source (bundler-style specifiers) with Node's native TypeScript support.
 *
 * @param {string} specifier
 * @param {import('node:module').ResolveHookContext} context
 * @param {import('node:module').ResolveFn} nextResolve
 */
export async function resolve(specifier, context, nextResolve) {
  if (!isRelativeSpecifier(specifier) || hasExtension(specifier) || !context.parentURL) {
    return nextResolve(specifier, context);
  }

  for (const ext of TS_EXTENSIONS) {
    const url = new URL(specifier + ext, context.parentURL);
    let filePath;
    try {
      filePath = fileURLToPath(url);
    } catch {
      continue;
    }
    if (existsSync(filePath) && statSync(filePath).isFile()) {
      return nextResolve(url.href, context);
    }
  }

  // e.g. ./dir -> ./dir/index.ts
  const dirUrl = new URL(specifier + "/", context.parentURL);
  try {
    const dirPath = fileURLToPath(dirUrl);
    if (existsSync(dirPath) && statSync(dirPath).isDirectory()) {
      for (const ext of TS_EXTENSIONS) {
        const indexUrl = new URL(`index${ext}`, dirUrl);
        const indexPath = fileURLToPath(indexUrl);
        if (existsSync(indexPath) && statSync(indexPath).isFile()) {
          return nextResolve(indexUrl.href, context);
        }
      }
    }
  } catch {
    // ignore
  }

  return nextResolve(specifier, context);
}
