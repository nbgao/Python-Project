from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed

def parse():
    filename = 'test1'
     # Read a bin file
    fn = open('./doc/' + filename + '.pdf', 'rb')
    # Create PDFParser
    parser = PDFParser(fn)
    # Create PDF document
    doc = PDFDocument()
    # Link parser & doc
    parser.set_document(doc)
    doc.set_parser(parser)

    doc.initialize("")
    # Detect document provide txt transfer
    if not doc.is_extractable:
        raise PDFTextExtractionNotAllowed
    else:
        # Create PDF resource manager
        resource = PDFResourceManager()
        # Create PDF params
        laparams = LAParams()
        # Create page aggregator to read PDF document
        device = PDFPageAggregator(resource, laparams=laparams)
        # Create interpreter to encode document
        interpreter = PDFPageInterpreter(resource, device)
        # Engodic pages
        # doc.get_pages() get page list

        for page in doc.get_pages():
            # Use interpreter read each page
            interpreter.process_page(page)
            # Use interpreter get content
            layout = device.get_result()
            # layout container objects from PDF
            for out in layout:
                if hasattr(out, "get_text"):
                    print(out.get_text)
                    with open('./doc/' + filename + '.txt', 'a', encoding='utf-8') as f:
                        f.write(out.get_text() + '\n')

if __name__ == '__main__':
    parse()