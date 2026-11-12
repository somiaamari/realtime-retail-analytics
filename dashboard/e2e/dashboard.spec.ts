import { expect, test } from "@playwright/test";

test("mock dashboard loads and global range selection is interactive", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Live operations" })).toBeVisible();
  await page.getByRole("link", { name: /Analytics/ }).click();
  await expect(page.getByRole("heading", { name: "Store analytics" })).toBeVisible();
  await page.getByLabel("Global date range").selectOption("168");
  await expect(page.getByRole("img", { name: /Entries, exits and average occupancy/ })).toBeVisible();
});

test("zone editor adds a polygon and saves changes", async ({ page }) => {
  await page.goto("/zones");
  await expect(page.getByRole("heading", { name: "Zones & lines" })).toBeVisible();
  await page.getByRole("button", { name: /Draw zone/ }).click();
  const canvas = page.getByRole("application", { name: "Interactive zone and line editor" });
  const box = await canvas.boundingBox();
  if (!box) throw new Error("Editor canvas is not visible");
  await page.mouse.click(box.x + box.width * .15, box.y + box.height * .25);
  await page.mouse.click(box.x + box.width * .3, box.y + box.height * .25);
  await page.mouse.click(box.x + box.width * .3, box.y + box.height * .5);
  await page.mouse.dblclick(box.x + box.width * .15, box.y + box.height * .5);
  await page.getByRole("button", { name: /Save changes/ }).click();
  await expect(page.getByText("All changes saved")).toBeVisible();
});
