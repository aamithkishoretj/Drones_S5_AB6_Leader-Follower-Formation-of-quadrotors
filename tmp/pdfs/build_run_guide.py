from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Preformatted, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/pdf/simulator_running_commands.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom', fontName='Helvetica-Bold', fontSize=23, leading=27, textColor=HexColor('#12384b'), spaceAfter=15))
styles.add(ParagraphStyle(name='Sub', fontName='Helvetica-Bold', fontSize=11, leading=14, spaceBefore=12, spaceAfter=6, textColor=HexColor('#126c79')))
styles.add(ParagraphStyle(name='BodyCustom', fontSize=9.5, leading=14, spaceAfter=7))
styles.add(ParagraphStyle(name='CodeCustom', fontName='Courier-Bold', fontSize=8.4, leading=12, backColor=HexColor('#eef3f6'), borderPadding=8, spaceBefore=5, spaceAfter=11))
story=[]
def p(text): story.append(Paragraph(text, styles['BodyCustom']))
def h(text): story.append(Paragraph(text, styles['Sub']))
def code(text):
    assert max(map(len,text.splitlines())) <= 105, text
    story.append(Preformatted(text, styles['CodeCustom']))
def page(n,title,terminal):
    if n>1: story.append(PageBreak())
    p(f'<b>RUN GUIDE / {n:02d}</b> &nbsp; | &nbsp; Updated 06 October 2026')
    story.append(Paragraph(title,styles['TitleCustom']))
    p(f'<b>Terminal:</b> {terminal}')
project='Dual-Quaternion-Based-Control-for-a-Leader-Follower-Formation-of-Two-Quadrotors'
win=f'cd "D:\\drones\\{project}"'
linux=f'cd /mnt/d/drones/{project}'
py=r'.\.venv\Scripts\python.exe'

page(1,'Gym-PyBullet: run the drones','Windows PowerShell')
p('This guide uses the existing project environments and installed simulators. The learned controller uses simulator-trained dynamics with dual-quaternion pose error. Run one simulation at a time.')
h('1. Open the project folder')
code(win)
h('2. Train only if needed')
p('The three backend models were generated in the previous update. Skip training when the matching model exists. Retrain after changing backend dynamics; training runs without a GUI.')
code(f'{py} -B scripts/train_data_driven.py `\n  --simulator pybullet')
h('3. Launch the figure-eight flight')
code(f'{py} -B scripts/run_experiment.py `\n  --simulator pybullet --controller learned `\n  --trajectory lemniscate --follow_distance 0.8 `\n  --duration 60 --gui')
p('The follower targets the leader\'s recorded path 0.8 m behind. Red trails show the leader; blue trails show the follower. Tracking has numerical error; the drones do not occupy the same position simultaneously.')
h('4. Stop or finish')
p('Wait for the configured duration, or press <b>Ctrl+C</b> in the terminal. An interrupted run may not save a result. Let the previous command finish before starting another simulation.')
p('<b>Copying commands:</b> PowerShell lines use a backtick (`) at the end for continuation. Paste the complete block; do not add spaces after the backtick. No environment activation is needed because the commands use its Python executable directly.')

page(2,'MuJoCo: run the drones','Windows PowerShell')
h('1. Open the project folder')
code(win)
h('2. Train if the MuJoCo model is missing')
code(f'{py} -B scripts/train_data_driven.py `\n  --simulator mujoco')
h('3. Launch the figure-eight flight')
code(f'{py} -B scripts/run_experiment.py `\n  --simulator mujoco --controller learned `\n  --trajectory lemniscate --follow_distance 0.8 `\n  --duration 60 --gui')
h('4. Run a B-spline trajectory instead')
code(f'{py} -B scripts/run_experiment.py `\n  --simulator mujoco --controller learned `\n  --trajectory bspline --follow_distance 0.8 `\n  --duration 60 --gui')
p('<b>Current limitation:</b> the MuJoCo backend stabilizes attitude but does not follow requested yaw. Its successful position tracking must not be described as full pose-tracking validation.')
h('Model files are simulator-specific')
p('MuJoCo uses <b>models/mujoco_response.npz</b>. PyBullet uses <b>models/pybullet_response.npz</b>. Do not copy or rename a model from another backend to bypass a missing-model error.')
p('For a run without a window, remove <b>--gui</b>. To hide the red/blue path lines, add <b>--no-trails</b>. Press <b>Ctrl+C</b> to stop early.')

page(3,'Gazebo: run through WSL','PowerShell first, then Ubuntu / WSL')
h('1. Enter Ubuntu from Windows PowerShell')
code('wsl -d Ubuntu-24.04')
p('Run every command below in the Ubuntu shell that opens, not in PowerShell.')
h('2. Open the project and load ROS + Python')
code(linux+'\nsource /opt/ros/jazzy/setup.bash\nsource .venv-gazebo/bin/activate')
h('3. Train if the Gazebo model is missing')
code('python -B scripts/train_data_driven.py --simulator gazebo')
h('4. Launch with the WSL graphics settings')
code('env -u QT_QUICK_BACKEND QT_QPA_PLATFORM=xcb \\\n  LIBGL_ALWAYS_SOFTWARE=1 \\\n  python -B scripts/run_experiment.py \\\n  --simulator gazebo --controller learned \\\n  --trajectory lemniscate --follow_distance 0.8 \\\n  --duration 60 --gui --gazebo_render_engine ogre')
h('5. If the GUI is black, check a headless run')
code('python -B scripts/run_experiment.py \\\n  --simulator gazebo --controller learned \\\n  --trajectory lemniscate --follow_distance 0.8 --duration 60')
p('Gazebo and ROS 2 Jazzy must already be installed. Headless flight was verified; WSL GUI rendering was not reverified. Runtime log paths are printed in the terminal. A successful headless run does not prove the graphics window is working.')
p('Ubuntu multiline commands use a backslash (\\), not a PowerShell backtick. Repeat step 2 in every new Ubuntu terminal. Press <b>Ctrl+C</b> to stop the current run.')

page(4,'Trajectories and control options','Windows PowerShell examples; start from the project folder')
p('These examples use PyBullet. Replace <b>pybullet</b> with <b>mujoco</b> to use its trained model. For Gazebo, change the same flags in page 3\'s Ubuntu command and keep its graphics prefix.')
h('3D potato-chip route')
code(f'{py} -B scripts/run_experiment.py `\n  --simulator pybullet --controller learned `\n  --trajectory potato_chip --duration 60 --gui')
h('B-spline route')
code(f'{py} -B scripts/run_experiment.py `\n  --simulator pybullet --controller learned `\n  --trajectory bspline --duration 60 --gui')
h('LERP / SLERP pose interpolation')
code(f'{py} -B scripts/run_experiment.py `\n  --simulator pybullet --controller learned `\n  --trajectory interpolated --duration 30 --gui')
h('Original analytic controller for comparison')
code(f'{py} -B scripts/run_experiment.py `\n  --simulator pybullet --controller analytic --experiment real_eig `\n  --trajectory lemniscate --duration 60 --gui')
p('Analytic mode needs no trained model and is not the data-driven controller. It still uses recorded-path following unless you explicitly change the formation mode.')
h('Useful flags to add or replace')
p('<b>--follow_distance 1.0</b>: one-metre gap along the recorded route.<br/><b>--duration 30</b>: shorter run. <b>--no-trails</b>: hide path lines.<br/><b>--output results/demo</b>: save this run in a separate folder.<br/><b>--trajectory_file configs/course_bspline.json</b>: use this B-spline definition.<br/><b>--kalman --position_noise_std 0.03</b>: optional noisy-position experiment.<br/><b>--optimize_path</b>: optional B-spline smoothing; needs CVXPY and --trajectory bspline.')
p('These are runnable options, not a guarantee of accuracy on every route. Keep control at 48 Hz and physics at 240 Hz for the supplied models. Changing these or backend dynamics requires compatible training.')

page(5,'Training, results and checks','PowerShell unless a block says Ubuntu')
h('Refit existing recordings without recollecting flights')
code(f'{py} -B scripts/train_data_driven.py `\n  --simulator pybullet --reuse-data')
p('Use matching duration and episode settings. Replace pybullet with mujoco for that backend. In Ubuntu, use <b>python -B</b> and <b>--simulator gazebo</b>. This command reruns evaluation and updates the model/report.')
h('Check tests and display every available run option')
code(f'{py} -B -m pytest -q\n{py} -B scripts/run_experiment.py --help')
h('Plot your latest PyBullet run')
code('$run = Get-ChildItem results/pybullet_learned_*.npz |\n  Sort-Object LastWriteTime -Descending | Select-Object -First 1\n'+f'{py} -B scripts/plot_results.py --run $run.FullName')
p('First complete a PyBullet run with the default results folder. Replace the filename pattern with <b>mujoco_learned_*.npz</b> for MuJoCo.')
h('Optional ideal simulation and course analysis')
code(f'{py} -B scripts/run_experiment.py `\n  --simulator kinematic --controller analytic --duration 30\n{py} -B scripts/analyze_course_concepts.py')
p('Kinematic mode is an ideal control-law check with no physics-engine GUI. The analysis script provides supporting mathematical demonstrations.')
h('Where your files are saved')
p('<b>models/</b>: simulator-specific trained response models.<br/><b>data/identification/&lt;backend&gt;/</b>: training recordings, validation flights and report.json.<br/><b>results/</b>: completed experiment .npz logs.<br/><b>results/learned_validation/summary.json</b>: saved 60-second validation metrics.<br/><b>docs/data_driven.md</b>: method, results and limitations.')
h('If a command fails')
p('<b>Trained model missing:</b> run the training command for that exact simulator.<br/><b>Cannot find Python or scripts:</b> repeat the folder command on the simulator\'s page and check the correct terminal.<br/><b>Gazebo ROS import error:</b> repeat both source commands in Ubuntu.<br/><b>Model timestep mismatch:</b> remove custom frequency flags to use the supplied model settings.<br/><b>Module not found:</b> the required backend dependency is absent from that environment; this guide assumes the existing installation.')

def footer(canvas,doc):
    canvas.setStrokeColor(HexColor('#cad7de')); canvas.line(40,39,A4[0]-40,39)
    canvas.setFont('Helvetica',8); canvas.setFillColor(HexColor('#536675'))
    canvas.drawString(40,26,'DUAL-QUATERNION LEADER / FOLLOWER  |  COMMAND REFERENCE')
    canvas.drawRightString(A4[0]-40,26,str(doc.page))
doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=40,leftMargin=40,topMargin=38,bottomMargin=51)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)

