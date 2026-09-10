-- tools/git-lastmod.lua
-- Adds below the title a line such as
-- "Last updated: 10 September 2026".
-- HTML only, for every .qmd file.

local months = {
  ["01"] = "January", ["02"] = "February", ["03"] = "March",
  ["04"] = "April",   ["05"] = "May",      ["06"] = "June",
  ["07"] = "July",    ["08"] = "August",   ["09"] = "September",
  ["10"] = "October", ["11"] = "November", ["12"] = "December"
}

local function capture(cmd)
  local h = io.popen(cmd)
  if not h then return nil end

  local out = h:read("*a")
  h:close()

  if not out then return nil end

  out = out:gsub("%s+$", "")
  return out ~= "" and out or nil
end

local function shell_quote(s)
  return "'" .. s:gsub("'", "'\\''") .. "'"
end

local function format_date_iso(d)
  local y, m, day = d:match("^(%d%d%d%d)%-(%d%d)%-(%d%d)$")
  if not (y and m and day) then return d end

  return tostring(tonumber(day)) .. " " .. (months[m] or m) .. " " .. y
end

local function is_qmd(path)
  local p = path:gsub("\\", "/")
  return p:match("%.qmd$") ~= nil
end

function Pandoc(doc)
  local fmt = FORMAT or ""

  if fmt:find("latex") or fmt:find("pdf") or fmt:find("beamer") then
    return doc
  end

  local input = quarto and quarto.doc and quarto.doc.input_file
  if not input then return doc end
  if not is_qmd(input) then return doc end

  local toplevel = capture("git rev-parse --show-toplevel 2>/dev/null")
  if not toplevel then return doc end

  local iso = capture(
    "git -C " .. shell_quote(toplevel) ..
    " log -1 --format=%cs -- " .. shell_quote(input) ..
    " 2>/dev/null"
  )

  if not (iso and iso:match("^%d%d%d%d%-%d%d%-%d%d$")) then
    return doc
  end

  local div = pandoc.Div(
    {
      pandoc.Para({
        pandoc.Emph({
          pandoc.Str("Last updated: " .. format_date_iso(iso))
        })
      })
    },
    pandoc.Attr("", {"lastmod"}, {})
  )

  table.insert(doc.blocks, 1, div)

  return doc
end