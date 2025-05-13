from fpdf import FPDF
import plotly.io as pio
import tempfile
import os

__all__ = [
        'PDF_Report'
]

class PDF_Report(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)
        self.set_font('Courier', '', 10)


    def header(self):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(40, 40, 40)
        self.cell(0, 10, 'Model Evaluation Report', ln=True, align='C')
        self.ln(5)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(5)


    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', align='R')


    def main_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.set_text_color(0)
        self.cell(0, 10, title.upper(), ln=True, fill=True, align='C')
        self.ln(3)


    def chapter_title(self, title):
        self.set_font('Arial', 'B', 10)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(0)
        self.cell(0, 10, title.upper(), ln=True, fill=True)
        self.ln(3)


    def chapter_body(self, text):
        self.set_font('Arial', '', 10)
        self.set_text_color(20)
        self.multi_cell(0, 3, text)
        self.ln()


    def chapter_df(self, text):
        self.set_font('Courier', '', 8)
        self.set_text_color(20)
        self.multi_cell(0, 3, text)
        self.ln()


    def add_image(self, img_path, w=160):
        # center the image on the page
        page_width = self.w - 2 * self.l_margin
        x = (page_width - w) / 2 + self.l_margin
        self.image(img_path, x=x, w=w)
        self.ln(3)


    @staticmethod
    def save_plotly_to_image(fig):
        tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        pio.write_image(fig, tmp_file.name, format='png')
        return tmp_file.name
