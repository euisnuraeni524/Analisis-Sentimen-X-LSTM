from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
import datetime as dt
import os




def generate_history_pdf(output_path: str, title: str, period_text: str, analysis_date: str,
                        n_rows: int, label_counts: dict, accuracy: float | None,
                        notes: str | None = None):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []


    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 12))


    meta = [
        ['Periode Data', period_text],
        ['Tanggal Analisis', analysis_date],
        ['Jumlah Data', str(n_rows)],
        ['Akurasi Model', f"{accuracy:.4f}" if accuracy is not None else '-'],
    ]
    t = Table(meta, hAlign='LEFT', colWidths=[120, 350])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eeeeee')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
    ]))
    story.append(t)
    story.append(Spacer(1, 12))


    if label_counts:
        rows = [['Label', 'Jumlah']]
        for k,v in label_counts.items():
            rows.append([k, str(v)])
        t2 = Table(rows, hAlign='LEFT', colWidths=[120, 120])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eeeeee')),
            ('BOX', (0,0), (-1,-1), 0.5, colors.black),
            ('INNERGRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
        ]))
        story.append(Paragraph('Distribusi Label', styles['Heading2']))
        story.append(t2)
        story.append(Spacer(1, 12))


    if notes:
        story.append(Paragraph('Catatan', styles['Heading2']))
        story.append(Paragraph(notes, styles['BodyText']))


    doc.build(story)
    return output_path