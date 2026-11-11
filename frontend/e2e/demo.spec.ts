import { expect, test } from '@playwright/test'
import path from 'node:path'

// Integration: frontend -> API -> real model inference -> results, following the demo script.
// Every step saves a screenshot to evidence/screenshots as genuine run evidence.
const shots = path.resolve(import.meta.dirname, '../../evidence/screenshots')
const shot = (name: string) => path.join(shots, `${name}.png`)
const samples = path.resolve(import.meta.dirname, '../../samples')

test.describe.configure({ mode: 'serial' })

test('01 backend health and system status', async ({ page, request }) => {
  const health = await request.get('/health')
  expect(health.ok()).toBeTruthy()
  const body = await health.json()
  expect(body.status).toBe('ok')
  expect(body.models_available).toBe(4)
  await page.goto('/settings')
  await expect(page.getByText('Backend health')).toBeVisible()
  await expect(page.getByText(/Device in use/)).toBeVisible()
  await page.screenshot({ path: shot('01_system_status'), fullPage: true })
})

test('02 sample analysis with ground truth: Swin transformer', async ({ page }) => {
  await page.goto('/new')
  await page.getByRole('tab', { name: /bundled samples/i }).click()
  await page.screenshot({ path: shot('02_sample_selection'), fullPage: true })
  await page.getByRole('button', { name: /dense change/i }).click()
  await page.getByRole('radio', { name: /^Siamese Swin Transformer LEVIR-CD/ }).click()
  await page.getByRole('button', { name: /run change detection/i }).click()

  await expect(page).toHaveURL(/\/results\/[0-9a-f]{32}$/)
  await expect(page.getByTestId('stat-F1-score')).toBeVisible({ timeout: 90_000 })
  const f1 = Number.parseFloat((await page.getByTestId('stat-F1-score').innerText()).replace('%', ''))
  const iou = Number.parseFloat((await page.getByTestId('stat-IoU').innerText()).replace('%', ''))
  expect(f1).toBeGreaterThan(50) // a trained model, not random weights
  expect(iou).toBeGreaterThan(40)
  await expect(page.getByTestId('stat-Changed pixels')).toContainText('%')
  await expect(page.getByRole('table', { name: 'Confusion matrix' })).toBeVisible()
  await page.waitForLoadState('networkidle')
  await page.screenshot({ path: shot('03_result_before_after_slider'), fullPage: true })

  for (const [tab, name] of [['Side by side', '04_side_by_side'], ['Overlay', '05_overlay'], ['Change mask', '06_change_mask'],
    ['Confidence', '07_confidence'], ['Ground truth', '08_ground_truth_comparison']] as const) {
    await page.getByRole('tab', { name: tab }).click()
    await page.waitForLoadState('networkidle')
    await page.screenshot({ path: shot(name), fullPage: true })
  }

  // threshold control re-thresholds on the server and the statistics change
  const before = await page.getByTestId('stat-Changed pixels').innerText()
  await page.locator('#res-threshold').fill('0.9')
  await expect(page.getByTestId('stat-Changed pixels')).not.toHaveText(before, { timeout: 15_000 })
  await page.screenshot({ path: shot('09_threshold_adjusted'), fullPage: true })

  // downloads
  for (const label of ['Change mask (PNG)', 'Overlay (PNG)', 'Results (JSON)']) {
    const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: label }).click()])
    expect(download.suggestedFilename()).toMatch(/^terravision_[0-9a-f]{8}_/)
  }
  await expect(page.getByRole('link', { name: 'Change mask (GeoTIFF)' })).toHaveCount(0) // not georeferenced

  // CNN vs transformer on the same pair
  await page.getByRole('button', { name: /Re-run with FC-Siam-diff/ }).click()
  await expect(page.getByText('FC-Siam-diff (CNN baseline)').first()).toBeVisible({ timeout: 90_000 })
  await expect(page.getByTestId('stat-F1-score')).toBeVisible({ timeout: 90_000 })
  await page.waitForLoadState('networkidle')
  await page.screenshot({ path: shot('10_cnn_rerun_same_pair'), fullPage: true })
})

test('03 model comparison shows the real benchmark and live sample evaluation', async ({ page }) => {
  await page.goto('/compare')
  await expect(page.getByRole('heading', { name: /LEVIR-CD — 0.5 m/ })).toBeVisible()
  await expect(page.getByRole('heading', { name: /OSCD — real Sentinel-2/ })).toBeVisible()
  await expect(page.getByText(/Domain shift/)).toBeVisible()
  await expect(page.getByText('Benchmark table')).toHaveCount(2)
  await page.getByRole('button', { name: /run now/i }).click()
  await expect(page.getByText(/OSCD samples/).first()).toBeVisible({ timeout: 120_000 })
  await expect(page.getByText(/LEVIR-CD samples/).first()).toBeVisible()
  await page.waitForTimeout(800)
  await page.screenshot({ path: shot('11_model_comparison'), fullPage: true })
})

test('04 manual upload of a GeoTIFF pair gives georeferenced results on the map', async ({ page }) => {
  await page.goto('/new')
  const inputs = page.locator('input[type=file]')
  await inputs.nth(0).setInputFiles(path.join(samples, 'sentinel2/bengaluru_north_t1.tif'))
  await inputs.nth(1).setInputFiles(path.join(samples, 'sentinel2/bengaluru_north_t2.tif'))
  await page.getByRole('button', { name: /run change detection/i }).click()
  await expect(page.getByTestId('stat-Changed pixels')).toBeVisible({ timeout: 90_000 })
  await expect(page.getByText('EPSG:32643')).toBeVisible()
  // automatic model choice: 10 m GeoTIFFs get the Sentinel-2 model, so no domain-mismatch warning
  await expect(page.getByText('Siamese Swin Transformer (Sentinel-2)').first()).toBeVisible()
  await expect(page.getByText(/out-of-distribution/)).toHaveCount(0)
  await expect(page.getByText(/No ground-truth mask was provided/)).toBeVisible()
  await expect(page.getByTestId('stat-F1-score')).toHaveCount(0)
  await page.getByRole('tab', { name: 'Map' }).click()
  await expect(page.locator('.leaflet-container')).toBeVisible()
  await page.waitForTimeout(2500)
  await page.screenshot({ path: shot('12_geotiff_result_on_map'), fullPage: true })
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: 'Change mask (GeoTIFF)' }).click()])
  expect(download.suggestedFilename()).toMatch(/mask\.tif$/)
})

test('05 error handling: mismatched image sizes are rejected with a clear message', async ({ page }) => {
  await page.goto('/new')
  const inputs = page.locator('input[type=file]')
  await inputs.nth(0).setInputFiles(path.join(samples, 'levir/test_45_t1.png'))
  await inputs.nth(1).setInputFiles(path.join(samples, 'sentinel2/bengaluru_north_t2.preview.png'))
  await page.getByRole('button', { name: /run change detection/i }).click()
  await expect(page.getByText('The analysis could not be started')).toBeVisible()
  await expect(page.getByText('IMAGE_DIMENSION_MISMATCH')).toBeVisible()
  await page.screenshot({ path: shot('13_error_dimension_mismatch'), fullPage: true })

  await inputs.nth(1).setInputFiles({ name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('not an image') })
  await expect(page.getByText(/Unsupported file type/)).toBeVisible()
  await page.screenshot({ path: shot('14_error_unsupported_file'), fullPage: true })
})

test('06 history, overview and research pages reflect real state', async ({ page }) => {
  await page.goto('/history')
  await expect(page.getByRole('table')).toBeVisible()
  await expect(page.getByText('completed').first()).toBeVisible()
  await page.screenshot({ path: shot('15_analysis_history'), fullPage: true })
  await page.goto('/')
  await expect(page.getByTestId('stat-Total analyses')).not.toHaveText('0')
  await page.waitForLoadState('networkidle')
  await page.screenshot({ path: shot('16_overview_dashboard'), fullPage: true })
  await page.goto('/research')
  await expect(page.getByRole('heading', { name: 'Literature' })).toBeVisible()
  await page.screenshot({ path: shot('17_research_and_models'), fullPage: true })
})

test('07 satellite map: live Sentinel-2 search (online only)', async ({ page, request }) => {
  const probe = await request.post('/api/satellite/search', {
    data: { provider: 'earth_search', bbox: { west: 77.6, south: 13.09, east: 77.66, north: 13.14 }, start: '2019-01-01', end: '2019-03-31', max_cloud_cover: 20, limit: 3 },
  })
  test.skip(!probe.ok(), 'Satellite catalogue unreachable (offline) - optional online capability')
  await page.goto('/map')
  await expect(page.locator('.leaflet-container')).toBeVisible()
  await page.getByRole('button', { name: /draw area/i }).click()
  const box = (await page.locator('.leaflet-container').boundingBox())!
  await page.mouse.move(box.x + box.width * 0.45, box.y + box.height * 0.42)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * 0.56, box.y + box.height * 0.56, { steps: 8 })
  await page.mouse.up()
  await expect(page.getByText(/^AOI /)).toBeVisible()
  await page.getByRole('button', { name: /find candidate scenes/i }).click()
  await expect(page.getByText('Candidate scenes')).toBeVisible({ timeout: 60_000 })
  await page.waitForTimeout(1500)
  await page.screenshot({ path: shot('18_satellite_map_candidates'), fullPage: true })
})

test('08 real Sentinel-2 test city with ground truth: OSCD-trained transformer on the map', async ({ page }) => {
  await page.goto('/new')
  await page.getByRole('tab', { name: /bundled samples/i }).click()
  await page.getByRole('button', { name: /OSCD lasvegas/ }).click()
  // selecting a Sentinel-2 sample switches to the Sentinel-2 model automatically
  await expect(page.getByRole('radio', { name: /Siamese Swin Transformer \(Sentinel-2\)/ })).toHaveAttribute('aria-checked', 'true')
  await page.getByRole('button', { name: /run change detection/i }).click()
  await expect(page.getByTestId('stat-F1-score')).toBeVisible({ timeout: 90_000 })
  const f1 = Number.parseFloat((await page.getByTestId('stat-F1-score').innerText()).replace('%', ''))
  expect(f1).toBeGreaterThan(30) // measured against OSCD ground truth on a held-out test city
  await expect(page.getByText(/out-of-distribution/)).toHaveCount(0)
  await page.getByRole('tab', { name: 'Ground truth' }).click()
  await page.waitForLoadState('networkidle')
  await page.screenshot({ path: shot('19_oscd_lasvegas_ground_truth'), fullPage: true })
  await page.getByRole('tab', { name: 'Map' }).click()
  await expect(page.locator('.leaflet-container')).toBeVisible()
  await page.waitForTimeout(2500)
  await page.screenshot({ path: shot('20_oscd_lasvegas_on_map'), fullPage: true })
})

test('09 Bengaluru Sentinel-2 pair now produces detections with the Sentinel-2 model', async ({ page }) => {
  await page.goto('/new')
  await page.getByRole('tab', { name: /bundled samples/i }).click()
  await page.getByRole('button', { name: /north Bengaluru/ }).click()
  await page.getByRole('button', { name: /run change detection/i }).click()
  await expect(page.getByTestId('stat-Changed pixels')).toBeVisible({ timeout: 90_000 })
  const changed = Number.parseFloat((await page.getByTestId('stat-Changed pixels').innerText()).replace('%', ''))
  expect(changed).toBeGreaterThan(0)
  await expect(page.getByText(/out-of-distribution/)).toHaveCount(0)
  await page.getByRole('tab', { name: 'Map' }).click()
  await page.waitForTimeout(2500)
  await page.screenshot({ path: shot('21_bengaluru_sentinel2_model_on_map'), fullPage: true })
})
