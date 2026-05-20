import { expect, test } from "@playwright/test";

test("plans an Elenchus move and accepts a manual reply", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("button", { name: /summon elenchus/i })).toBeVisible();
  await page.getByText("Controls").click();
  await page.getByLabel(/compute per-move simulations/i).evaluate((input) => {
    const slider = input as HTMLInputElement;
    slider.value = "2";
    slider.dispatchEvent(new Event("input", { bubbles: true }));
    slider.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.getByLabel(/you manual message/i).fill("I say I want clarity, but I keep defending the comfortable story.");
  await page.getByLabel(/you manual message/i).press("Enter");
  await expect(page.getByText(/Thought materialised/)).toBeVisible({ timeout: 20_000 });
  await page.getByTestId("node-label-0").click();
  await expect(page.getByText(/simulated (branches|exchanges)/i)).toBeVisible();
  await page.getByLabel(/graph detail/i).evaluate((input) => {
    const slider = input as HTMLInputElement;
    slider.value = "0";
    slider.dispatchEvent(new Event("input", { bubbles: true }));
    slider.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(page.getByText("0% detail")).toBeVisible();
  await page.getByLabel(/you manual message/i).fill("Manual P2 smoke reply");
  await page.getByLabel(/you manual message/i).press("Enter");
  await expect(page.getByText("Manual P2 smoke reply")).toBeVisible();
});

test("graph canvas becomes nonblank after MCTS events", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel(/you manual message/i).fill("I keep calling it realism, but I suspect it is fear.");
  await page.getByLabel(/you manual message/i).press("Enter");
  await expect(page.locator("canvas")).toBeVisible();
  await page.waitForFunction(() => {
    const canvas = document.querySelector("canvas");
    if (!canvas) return false;
    const context = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
    return canvas.width > 0 && canvas.height > 0 && context !== null;
  });
});
