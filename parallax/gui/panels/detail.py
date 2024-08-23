from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QTableWidget,
    QTableWidgetItem, QLineEdit, QPlainTextEdit, QPushButton,
    QInputDialog, QHeaderView,
)
from PyQt6.QtCore import Qt, pyqtSignal

_CLS_COLORS = {
    "unverified": "#c0392b",
    "known": "#a8d8ea",
}


class DetailPanel(QWidget):
    candidate_closed = pyqtSignal()
    candidate_updated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        outer.addWidget(self._scroll)

        self._inner = QWidget()
        self._layout = QVBoxLayout(self._inner)
        self._scroll.setWidget(self._inner)

        self._current_id = None
        self.show_idle()

    def _clear_layout(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def show_idle(self, _from_button=False):
        self._clear_layout()
        self._current_id = None
        if _from_button:
            self.candidate_closed.emit()

    def show_hint(self):
        self._clear_layout()
        self._current_id = None
        self._layout.addWidget(QLabel("Select a candidate"))

    def load(self, candidate_id):
        self._clear_layout()
        self._current_id = candidate_id

        try:
            from parallax import catalog
            cand = catalog.get(candidate_id)
        except Exception:
            cand = None

        if cand is None:
            self._layout.addWidget(QLabel("Candidate not found"))
            return

        close_row = QHBoxLayout()
        close_row.addStretch()

        is_bookmarked = "bookmarked" in (cand.tags or [])
        bm_btn = QPushButton("Bookmarked" if is_bookmarked else "Bookmark")
        if is_bookmarked:
            bm_btn.setStyleSheet(
                "background-color: #f1c40f; color: #333; font-size: 11px; padding: 2px 6px;"
            )
        else:
            bm_btn.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        bm_btn.clicked.connect(lambda: self._toggle_bookmark(candidate_id, is_bookmarked))
        close_row.addWidget(bm_btn)

        close_btn = QPushButton("X")
        close_btn.setFixedSize(22, 22)
        close_btn.setStyleSheet("font-size: 11px; padding: 0;")
        close_btn.clicked.connect(lambda: self.show_idle(_from_button=True))
        close_row.addWidget(close_btn)
        row_w = QWidget()
        row_w.setLayout(close_row)
        self._layout.addWidget(row_w)

        self._layout.addWidget(QLabel(f"ID: {cand.id}"))
        color = _CLS_COLORS.get(cand.classification, "#000000")
        cls_label = QLabel(cand.classification)
        cls_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self._layout.addWidget(cls_label)
        self._layout.addWidget(QLabel(f"RA: {cand.ra:.6f}"))
        self._layout.addWidget(QLabel(f"Dec: {cand.dec:.6f}"))
        self._layout.addWidget(QLabel(f"SNR: {cand.snr:.2f}"))

        conf = cand.confidence
        if conf >= 0.75:
            quality = "High"
            quality_color = "#27ae60"
        elif conf >= 0.50:
            quality = "Medium"
            quality_color = "#f39c12"
        else:
            quality = "Low"
            quality_color = "#c0392b"
        conf_label = QLabel(f"Detection quality: {quality}  ({conf:.2f})")
        conf_label.setStyleSheet(f"color: {quality_color};")
        conf_label.setToolTip(
            "Detection quality reflects how well this source was measured:\n"
            "SNR, number of filters it was detected in, distance to the\n"
            "nearest catalog source, and flux measurement reliability.\n\n"
            "High = well-measured, multiple filters, isolated\n"
            "Medium = adequately measured\n"
            "Low = marginal detection, single filter, or near a known source\n\n"
            "This score does NOT indicate astrophysical significance.\n"
            "A faint jet knot scores lower than a bright artifact."
        )
        self._layout.addWidget(conf_label)

        _AUTO_TAGS = {
            "narrowband_only", "line_dominated", "compact",
            "extended", "isolated", "crowded", "near_emission",
        }
        auto = [t for t in (cand.tags or []) if t in _AUTO_TAGS]
        if auto:
            tags_label = QLabel("Flags: " + "  ".join(auto))
            tags_label.setStyleSheet("color: #888; font-size: 10px;")
            self._layout.addWidget(tags_label)

        if cand.detections:
            self._layout.addWidget(QLabel("Detections"))
            tbl = QTableWidget(len(cand.detections), 5)
            tbl.setHorizontalHeaderLabels(["Filter", "SNR", "Flux", "Flux(MJy)", "Mag(AB)"])
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            tbl.setMaximumHeight(30 + 25 * len(cand.detections))
            dets_sorted = sorted(cand.detections, key=lambda d: d.snr, reverse=True)
            for i, det in enumerate(dets_sorted):
                tbl.setItem(i, 0, QTableWidgetItem(det.filter))
                tbl.setItem(i, 1, QTableWidgetItem(f"{det.snr:.2f}"))
                tbl.setItem(i, 2, QTableWidgetItem(f"{det.flux:.2f}"))
                fmjy = f"{det.flux_mjy:.3e}" if det.flux_mjy is not None else "-"
                mab = f"{det.mag_ab:.2f}" if det.mag_ab is not None else "-"
                tbl.setItem(i, 3, QTableWidgetItem(fmjy))
                tbl.setItem(i, 4, QTableWidgetItem(mab))
            self._layout.addWidget(tbl)

        if cand.catalog_matches:
            self._layout.addWidget(QLabel("Catalog Matches"))
            tbl = QTableWidget(len(cand.catalog_matches), 4)
            tbl.setHorizontalHeaderLabels(["Catalog", "ID", "Sep(\")", "Type"])
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            tbl.setMaximumHeight(30 + 25 * len(cand.catalog_matches))
            for i, m in enumerate(cand.catalog_matches):
                tbl.setItem(i, 0, QTableWidgetItem(m.catalog))
                tbl.setItem(i, 1, QTableWidgetItem(m.source_id))
                tbl.setItem(i, 2, QTableWidgetItem(f"{m.separation_arcsec:.2f}"))
                tbl.setItem(i, 3, QTableWidgetItem(m.object_type or ""))
            self._layout.addWidget(tbl)

        self._layout.addWidget(QLabel("Tags"))
        self._tags_input = QLineEdit(", ".join(cand.tags) if cand.tags else "")
        self._tags_input.editingFinished.connect(self._on_tags_changed)
        self._layout.addWidget(self._tags_input)

        self._layout.addWidget(QLabel("Notes"))
        self._notes_view = QPlainTextEdit()
        self._notes_view.setPlainText("\n".join(cand.notes) if cand.notes else "")
        self._notes_view.setReadOnly(True)
        self._notes_view.setMaximumHeight(100)
        self._layout.addWidget(self._notes_view)

        btn = QPushButton("Add note")
        btn.clicked.connect(self._on_add_note)
        self._layout.addWidget(btn)

        self._layout.addStretch()

    def _toggle_bookmark(self, candidate_id, currently_bookmarked):
        try:
            from parallax import archive
            if currently_bookmarked:
                archive.unbookmark(candidate_id)
            else:
                archive.bookmark(candidate_id)
        except Exception:
            pass
        self.candidate_updated.emit(self._current_id)
        self.load(self._current_id)

    def _on_tags_changed(self):
        if not self._current_id:
            return
        raw = self._tags_input.text()
        tags = [t.strip() for t in raw.split(",") if t.strip()]
        try:
            from parallax import archive
            archive.tag(self._current_id, tags)
        except Exception:
            pass
        self.load(self._current_id)

    def _on_add_note(self):
        if not self._current_id:
            return
        text, ok = QInputDialog.getText(self, "Add note", "Note:")
        if ok and text.strip():
            try:
                from parallax import archive
                archive.annotate(self._current_id, text.strip())
            except Exception:
                pass
            self.load(self._current_id)

    def clear(self):
        self.show_idle()
