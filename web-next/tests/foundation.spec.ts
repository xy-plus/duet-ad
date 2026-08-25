import { expect, test } from '@playwright/test';

test('boots the governed UI shell', async ({ page }) => {
  await page.goto('/');

  await expect(page).toHaveTitle('Duet AI 视频工作台');
  await expect(page.getByText('前端基础层已就绪')).toBeVisible();
});
