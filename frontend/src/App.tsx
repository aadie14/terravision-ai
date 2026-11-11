import { Compass } from 'lucide-react'
import { Link, Route, Routes } from 'react-router-dom'

import { Layout } from './components/Layout'
import { ToastProvider } from './components/Toasts'
import { EmptyState } from './components/ui'
import { ComparePage } from './pages/Compare'
import { HistoryPage } from './pages/History'
import { MapPage } from './pages/MapPage'
import { NewAnalysisPage } from './pages/NewAnalysis'
import { OverviewPage } from './pages/Overview'
import { ResearchPage } from './pages/Research'
import { ResultsIndexPage, ResultsPage } from './pages/Results'
import { SettingsPage } from './pages/Settings'

function NotFound() {
  return (
    <EmptyState icon={<Compass className="h-8 w-8" />} title="Page not found" action={<Link className="text-accent-300 underline" to="/">Back to overview</Link>}>
      This address does not match any TerraVision page.
    </EmptyState>
  )
}

export default function App() {
  return (
    <ToastProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<OverviewPage />} />
          <Route path="new" element={<NewAnalysisPage />} />
          <Route path="map" element={<MapPage />} />
          <Route path="results" element={<ResultsIndexPage />} />
          <Route path="results/:id" element={<ResultsPage />} />
          <Route path="compare" element={<ComparePage />} />
          <Route path="history" element={<HistoryPage />} />
          <Route path="research" element={<ResearchPage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </ToastProvider>
  )
}
