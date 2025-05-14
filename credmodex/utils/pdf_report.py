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
        self.reference_name_page = 'Model'


    def header(self):
        self.set_font('Arial', 'B', 11)
        self.set_text_color(40, 40, 40)
        self.cell(0, 10, f'{self.reference_name_page} Evaluation Report', ln=True, align='C')
        self.ln(1.5)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(1.5)


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
        page_width = self.w - 2 * self.l_margin
        x = (page_width - w) / 2 + self.l_margin
        self.image(img_path, x=x, w=w)
        self.ln(3)


    def add_chapter_model_page(self, text):
        self.add_page()

        # Draw full-page grey background
        self.set_fill_color(20, 20, 20)  # Light grey
        self.rect(0, 0, self.w, self.h, style='F')

        # Add centered text
        self.set_font('COurier', 'B', 52)
        self.set_text_color(240, 240, 240)

        # Calculate vertical center
        page_height = self.h
        page_width = self.w
        self.set_xy(0, page_height / 2 - 10)
        self.cell(page_width, 20, text, align='C')
        self.reference_name_page = text


    def add_chapter_rating_page(self, text1, text2):
        self.add_page()

        # Background
        self.set_fill_color(190, 190, 190)
        self.rect(0, 0, self.w, self.h, style='F')

        page_width = self.w
        page_height = self.h

        # Fonts and spacing
        font_large = 40
        font_small = 21
        line_spacing = 4  # small vertical space in layout units

        # Measure actual vertical height used (visually)
        text1_height = 14  # estimate: ~font_size * 0.35
        text2_height = 8   # estimate: ~font_size * 0.38

        total_height = text1_height + line_spacing + text2_height
        y_start = (page_height - total_height) / 2

        # Draw first line
        self.set_font('Courier', 'B', font_large)
        self.set_text_color(20, 20, 20)
        self.set_xy(0, y_start)
        self.cell(page_width, text1_height, text1, align='C')

        # Draw second line
        self.set_font('Courier', 'B', font_small)
        self.set_xy(0, y_start + text1_height + line_spacing)
        self.cell(page_width, text2_height, text2, align='C')

        self.reference_name_page = f'{text1} ({text2})'



    @staticmethod
    def save_plotly_to_image(fig):
        tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        pio.write_image(fig, tmp_file.name, format='png')
        return tmp_file.name
