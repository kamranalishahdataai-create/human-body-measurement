# Video Narration Script — Body Measurement AI Demo

Use this script while screen-recording the Streamlit demo. Read each section while showing the corresponding part of the interface.

---

## Intro (Show the main page)
> "This is our Body Measurement AI system. It automatically extracts 31 body measurements from a 3D body scan — things like chest circumference, waist, hip, shoulder width, sleeve length, and more — all with 96.3% accuracy. No tape measure needed."

---

## Step 1: Select a Subject (Click the dropdown)
> "On the left sidebar, we have our settings. In the center, we select a subject from our database. We have 36 test subjects, each scanned with a professional 3D body scanner. Let me pick one."

*(Select a subject from dropdown)*

---

## Step 2: Extract Measurements (Click the red button)
> "Now I click 'Extract Measurements'. The AI reads the 3D body scan file — which has 35,490 surface points — and calculates all body measurements in about one second."

*(Click the Extract Measurements button)*

---

## Step 3: Results Summary (Point to the 4 metric boxes)
> "At the top here we see the summary: 31 measurements extracted, 96.3% accuracy, the person's height in centimeters, and the method used — 3D Scan. This accuracy comes from our calibration system trained on all 36 subjects."

---

## Step 4: Measurements Tab (Scroll through the cards)
> "In the Measurements tab, all 31 measurements are organized by category — Core Dimensions, Upper Body Circumferences, Torso, Lower Body, and Lengths."

> "Each card shows the measurement name in Chinese and English, the AI prediction in centimeters — that's the big blue number — and below it, the Ground Truth, which is the real value measured by hand with a tape measure. The number in brackets shows the error."

> "Green means very accurate, less than 1.5 centimeters off. Orange means good, under 3 cm. As you can see, most measurements are green — very close to the real values."

*(Scroll down slowly through the measurement cards)*

---

## Step 5: Charts Tab (Click Charts tab)
> "In the Charts tab, we see a visual comparison. The top chart shows all circumference and breadth measurements as horizontal bars."

> "Below that is the key comparison chart — blue bars are AI predictions, green bars are the actual tape-measure values. You can see they're very close to each other. The statistics at the bottom show the mean error and what percentage of measurements are within 2 centimeters."

*(Scroll to show the comparison chart)*

---

## Step 6: 3D Model Tab (Click 3D Model tab)
> "This is the actual 3D body model from the scanner. You can drag to rotate it, scroll to zoom in, and right-click to pan. This mesh has 35,490 vertices — every point on the body surface that our AI uses to calculate measurements."

*(Rotate the 3D model slowly)*

---

## Step 7: Export Tab (Click Export tab)
> "Finally, the Export tab lets you download all measurements as JSON for software integration, or CSV for Excel spreadsheets. This makes it easy to integrate into any clothing, tailoring, or fitness system."

---

## Step 8: Settings (Point to sidebar)
> "A few more things in the sidebar: 'Apply Calibration' improves accuracy from 95.2% to 96.3% — it's on by default and recommended. 'Show Ground Truth' shows the real measurements for comparison. And 'Show 3D Model' loads the interactive body visualization."

---

## Closing
> "To summarize: this system takes a 3D body scan, extracts 31 precise measurements in one second, with 96.3% accuracy — verified against real tape-measure values across 36 test subjects. It can be integrated into any clothing, tailoring, or fitness application via our API."

---

## Tips for Recording
- Use **OBS Studio** or **Windows Game Bar** (Win+G) to record your screen
- Keep the browser fullscreen for clean visuals
- Speak slowly and clearly
- Pause 2-3 seconds between sections
- The **Guide Mode toggle** (🎓 in sidebar) shows explanatory text on screen — turn it ON while recording so viewers can read along
