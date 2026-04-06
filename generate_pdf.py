import asyncio
from playwright.async_api import async_playwright
import os

async def generate_pdf():
    print("🎨 Generating your beautifully styled PDF Carousel...")
    
    html_path = f"file:///{os.path.abspath('carousel_slides.html').replace(chr(92), '/')}"
    output_pdf = "LinkedIn_Carousel_Final.pdf"

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Load the HTML template
        await page.goto(html_path, wait_until="networkidle")
        
        # Generate the PDF with the correct 1080x1080 dimensions for LinkedIn
        await page.pdf(
            path=output_pdf,
            print_background=True,
            width="1080px",
            height="1080px",
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"}
        )
        await browser.close()
        
    print(f"✅ Success! Your carousel has been saved to: {output_pdf}")
    print("When you are ready, replace the placeholder text in 'carousel_slides.html' with image tags pointing to your screenshots, and run this script again!")

if __name__ == "__main__":
    asyncio.run(generate_pdf())
