"""
FLIGHT PATH CALCULATOR — Interactive Slingshot Prototype (v2)
============================================================
An interactive 2D "gravity assist" mini-simulator. You launch a rocket
from Earth by clicking, holding, and dragging it like a slingshot —
pull back, release, and it flies off in the opposite direction with
speed proportional to how far you pulled.

Mars exerts REAL simplified gravity:
  - Too far away  -> Mars has no meaningful pull (out of its "sphere
                     of influence"), rocket flies in a straight line.
  - Within range  -> Mars bends the trajectory (this is the slingshot /
                     gravity-assist effect — the closer the flyby, the
                     stronger the bend and speed boost).
  - Too close     -> the rocket crashes into Mars. Mission failure.

NEW IN v2:
  - Two-leg missions: reach the target, then slingshot BACK to Earth.
    A successful return recovers/reuses the hardware, which knocks a
    chunk off your mission cost (this is the "save money" part).
  - A live velocity readout (current speed, updated every frame).
  - Mars and the Target are placed with a little random jitter each
    run, so the layout isn't identical every time — but they're
    guaranteed to be placed far enough apart (and from Earth) that
    they can't spawn overlapping or in an unwinnable configuration.
  - Launches are no longer an instant jump to full speed. The engines
    ramp up over ~2 seconds, so early flight feels slower/heavier —
    closer to how a real rocket actually leaves the pad — and fuel for
    that burn is spent gradually over the ramp instead of all at once.
  - Manual control: hold WASD/Arrow keys mid-flight to fire manual
    thrusters and steer the rocket directly. Controls are relative to
    the ship's own nose, like a pilot's seat: Up/W = thrust forward
    (the way the nose is pointing), Down/S = thrust backward (retro),
    Left/A and Right/D = strafe left/right of the nose. It works, but
    it's expensive — manual thrusting burns fuel and battery MUCH
    faster than just coasting.
  - Running out of fuel OR battery doesn't end the mission outright —
    the rocket just goes uncontrollable. No more manual thrusters, no
    more launches; it simply coasts on momentum and whatever gravity
    does to it from there. It can still crash, drift off into space,
    or get lucky and coast into the target/Earth on momentum alone.
  - Random events (solar flares, micrometeoroids, sensor faults, etc.)
    now actually DO something: they knock the rocket off its computed
    trajectory with a real velocity kick, throw up an on-screen alert,
    and require you to grab manual control and hold WASD/Arrows for a
    moment to correct course. Ignore the alert and it expires on its
    own — but leaves your risk score permanently higher for the rest
    of the mission. (If you're already out of fuel/power, you can't
    correct events at all — they just happen to you.)

Mission stats (fuel, battery, cost, risk, velocity) update LIVE, frame
by frame, as the flight happens — not just as a one-time report.

FIX (this version): gravity from Earth/Target/Mars is now "softened" —
see GRAVITY_SOFTENING below — so acceleration no longer blows up toward
infinity as distance approaches zero (an inverse-square singularity).
The rocket starts flight sitting essentially on top of Earth, and after
reaching the target it re-launches sitting essentially on top of the
Target, so without softening those bodies' own gravity could spike
velocity into the hundreds of thousands of units/s in a single frame.

Requires:  pip install pygame numpy      (or: py -m pip install pygame numpy)
Run:       python3 flight_path_calculator.py   (or: py flight_path_calculator.py)

Controls:
  Click + hold + drag the rocket, then release  -> launch (slingshot)
  W/Up, S/Down (hold, mid-flight)               -> thrust forward / backward (ship-relative)
  A/Left, D/Right (hold, mid-flight)             -> strafe left / right (ship-relative)
  R                                              -> reset / try again
  ESC or close window                            -> quit
"""

import asyncio
import math
import random
import numpy as np
import pygame

# ----------------------------------------------------------------------
# WORLD / PHYSICS CONSTANTS
# ----------------------------------------------------------------------
W, H = 1000, 620
EARTH_POS = (150, 480)          # fixed — always the launch / return-home point
BASE_MARS_POS = (520, 240)      # nominal Mars position (jittered each run)
BASE_TARGET_POS = (860, 420)    # nominal target position (jittered each run)

MARS_VISUAL_RADIUS = 26
CRASH_RADIUS = 46          # get this close to Mars -> crash
INFLUENCE_RADIUS = 260     # beyond this, Mars gravity is negligible
CAPTURE_RADIUS = 50        # get this close to target/Earth -> captured
G_MARS = 350000            # tuned gravity strength constant (game-feel, not real units)

# NOTE: Earth and the Target used to also exert their own gravity, but it
# tended to trap the rocket in a tight orbit / hold it in place near either
# body, so that's been removed — only Mars exerts gravity now. Capture is
# purely proximity-based (CAPTURE_RADIUS below).

# --- FIX: gravitational softening -------------------------------------
# Inverse-square gravity (accel = G / dist^2) blows up toward infinity as
# dist -> 0. Every outbound leg *starts* at EARTH_POS, and every return
# leg re-launches from right on top of the Target, so without a floor on
# distance, the very first frame(s) of flight could compute a near-zero
# distance and produce an enormous, physically-nonsensical velocity kick
# (this is what caused speeds like ~600,000 u/s at launch). Clamping the
# distance used in the 1/dist^2 term to a minimum value ("softening")
# keeps acceleration finite and game-feel intact everywhere outside that
# tiny radius, which is standard practice in N-body sims.
GRAVITY_SOFTENING = 40.0   # px — minimum distance used in gravity calcs

MIN_PULL = 18               # px — shorter drags are ignored (no launch)
MAX_PULL = 170              # px — pull is clamped here (max power)
MIN_LAUNCH_SPEED = 14       # px/s at MIN_PULL (commanded / final speed, not instant)
MAX_LAUNCH_SPEED = 62       # px/s at MAX_PULL

# --- launch ramp-up (v2): real rockets build up speed gradually, they
#     don't teleport to full velocity the instant the engines light ---
BOOST_RAMP_SECONDS = 2.4    # sim-seconds for engines to reach commanded speed
LIFTOFF_SPEED_FRAC = 0.10   # fraction of commanded speed present at ignition instant

MARS_TARGET_JITTER = 45     # px of random placement variation each run
MIN_MARS_TARGET_SEP = CRASH_RADIUS + CAPTURE_RADIUS + 90   # keep them from colliding
MIN_MARS_EARTH_SEP = INFLUENCE_RADIUS * 0.62                # keep Mars off the pad
MIN_TARGET_EARTH_SEP = 250                                  # keep some flight distance

HOURS_PER_SECOND = 85       # real seconds -> simulated mission hours (for battery drain)
FUEL_LAUNCH_BURN_FRAC = 0.30   # fraction of fuel budgeted per launch burn, scaled by power
FUEL_DRAIN_OVER_60S = 0.35     # fraction of remaining fuel drained continuously per ~60s of flight
OPS_COST_PER_SECOND = 450      # $/s ongoing mission operations cost while flying
MAX_MISSION_TIME = 200         # seconds before a still-flying mission is declared lost
RECOVERY_FRACTION = 0.35       # fraction of hardware cost recovered by a successful return-to-Earth

# --- manual control (v3): hold WASD/Arrows mid-flight to steer directly.
#     Much more capable than coasting, but MUCH thirstier for resources. ---
MANUAL_THRUST_ACCEL = 55.0       # px/s^2 of extra acceleration while manually steering
MANUAL_FUEL_DRAIN_MULT = 9.0     # manual thrusting burns fuel this many times faster than coasting
MANUAL_BATTERY_DRAIN_MULT = 5.0  # manual guidance/thrusters draw this many times more power

# --- random events now really perturb the trajectory (v3) ---
EVENT_CHANCE_PER_FRAME = 0.0024  # occurs noticeably more often than before
EVENT_KICK_BASE = 14.0           # px/s minimum sudden velocity kick from an event
EVENT_KICK_SCALE = 220.0         # extra px/s kick, scaled by the event's "impact" value
ALERT_GRACE_PERIOD = 8.0         # seconds the player has to manually correct before it's too late
MANUAL_CORRECTION_HOLD = 0.5     # seconds of manual input required to resolve an active alert
UNRESOLVED_EVENT_RISK_PENALTY = 0.09  # permanent risk added if an alert is ignored until it expires

G0 = 9.81


# ----------------------------------------------------------------------
# BASELINE MISSION CALCULATIONS (real rocket-equation math, used as the
# starting values that the live simulation then consumes/updates)
# ----------------------------------------------------------------------
def fuel_required(delta_v, isp, dry_mass):
    exhaust_velocity = isp * G0
    mass_ratio = math.exp(delta_v / exhaust_velocity)
    wet_mass = dry_mass * mass_ratio
    propellant_mass = wet_mass - dry_mass
    return propellant_mass, wet_mass


def battery_required(mission_hours, avg_power_draw_w, safety_margin=1.3):
    energy_wh = mission_hours * avg_power_draw_w * safety_margin
    battery_mass_kg = energy_wh / 150.0
    return energy_wh, battery_mass_kg


def base_hardware_cost(dry_mass, propellant_mass):
    return dry_mass * 25000 + propellant_mass * 40


RANDOM_EVENTS = [
    ("Solar flare / radiation spike", 0.08),
    ("Micrometeoroid strike", 0.10),
    ("Communication blackout", 0.04),
    ("Navigation/sensor fault", 0.07),
    ("Thermal system anomaly", 0.05),
    ("Engine misfire", 0.09),
    ("Guidance computer glitch", 0.06),
    ("Debris field encounter", 0.08),
    ("Reaction wheel stall", 0.06),
    ("Cosmic ray bit-flip", 0.05),
]


# ----------------------------------------------------------------------
# GAME
# ----------------------------------------------------------------------
class SlingshotGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Flight Path Simulator — Slingshot Prototype")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 19, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 14)
        self.big_font = pygame.font.SysFont("consolas", 30, bold=True)

        # --- baseline mission numbers (computed once, then consumed live) ---
        self.dry_mass = 1200
        self.isp = 320
        self.avg_power_draw = 350
        self.nominal_delta_v = 9000
        self.propellant_total, self.wet_mass = fuel_required(
            self.nominal_delta_v, self.isp, self.dry_mass)
        self.battery_total_wh, _ = battery_required(4400, self.avg_power_draw)
        self.hardware_cost = base_hardware_cost(self.dry_mass, self.propellant_total)

        # --- starfield ---
        rng = np.random.default_rng(3)
        self.stars = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(1, 2.6))
                      for _ in range(170)]

        self.reset()

    # ------------------------------------------------------------------
    def _pick_positions(self):
        """Jitter Mars and the Target a bit each run, but reject any
        layout where they'd overlap each other, sit inside Mars' crash
        zone, or spawn too close to Earth (keeps every run winnable)."""
        for _ in range(300):
            mars = (BASE_MARS_POS[0] + random.uniform(-MARS_TARGET_JITTER, MARS_TARGET_JITTER),
                    BASE_MARS_POS[1] + random.uniform(-MARS_TARGET_JITTER, MARS_TARGET_JITTER))
            target = (BASE_TARGET_POS[0] + random.uniform(-MARS_TARGET_JITTER, MARS_TARGET_JITTER),
                      BASE_TARGET_POS[1] + random.uniform(-MARS_TARGET_JITTER, MARS_TARGET_JITTER))
            d_mars_target = math.hypot(mars[0] - target[0], mars[1] - target[1])
            d_mars_earth = math.hypot(mars[0] - EARTH_POS[0], mars[1] - EARTH_POS[1])
            d_target_earth = math.hypot(target[0] - EARTH_POS[0], target[1] - EARTH_POS[1])
            if (d_mars_target > MIN_MARS_TARGET_SEP
                    and d_mars_earth > MIN_MARS_EARTH_SEP
                    and d_target_earth > MIN_TARGET_EARTH_SEP):
                return mars, target
        # extremely unlikely fallback — use the nominal, always-valid layout
        return BASE_MARS_POS, BASE_TARGET_POS

    # ------------------------------------------------------------------
    def reset(self):
        self.state = "aim"          # aim | flight | crashed | lost | returned
        self.leg = "outbound"       # outbound | return
        self.mars_pos, self.target_pos = self._pick_positions()

        self.pos = list(EARTH_POS)
        self.vel = [0.0, 0.0]
        self.speed_current = 0.0
        self.heading_deg = -90.0     # nose direction, degrees; used for visuals AND manual thrust
        self.dragging = False
        self.drag_mouse = None
        self.elapsed = 0.0
        self.fuel_remaining = self.propellant_total
        self.battery_remaining = self.battery_total_wh
        self.cost_current = self.hardware_cost
        self.risk_current = 0.15     # baseline complexity risk
        self.event_log = []          # list of [text, life_seconds]
        self.active_event_impacts = []  # list of [impact, remaining_decay]
        self.trail = []
        self.min_mars_dist_this_flight = 1e9
        self.recovery_savings = 0.0

        # launch-ramp state (set properly on each try_launch)
        self.launch_dir = (0.0, 0.0)
        self.launch_target_speed = 0.0
        self.launch_fuel_budget = 0.0
        self.boost_elapsed = 0.0
        self.boost_duration = 0.0

        # manual control + event-alert state
        self.manual_active = False       # True this frame if the player is holding thrust keys
        self.manual_wh_used = 0.0        # extra battery burned by manual thrusting (cumulative)
        self.manual_alert = None         # dict when an event has knocked us off course, else None
        self.unresolved_penalty = 0.0    # permanent risk added by ignored alerts
        self.out_of_resources = False    # True once fuel OR battery hits zero — ship goes uncontrollable

    # ------------------------------------------------------------------
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    self.reset()
            if event.type == pygame.MOUSEBUTTONDOWN and self.state == "aim" and not self.out_of_resources:
                mx, my = event.pos
                if math.hypot(mx - self.pos[0], my - self.pos[1]) < 55:
                    self.dragging = True
                    self.drag_mouse = (mx, my)
            if event.type == pygame.MOUSEMOTION and self.dragging:
                self.drag_mouse = event.pos
            if event.type == pygame.MOUSEBUTTONUP and self.dragging:
                self.dragging = False
                self.try_launch(event.pos)
        return True

    # ------------------------------------------------------------------
    def _launch_params(self, mouse_pos):
        """Shared by try_launch() and preview_path() so the ghost trajectory
        is computed with EXACTLY the same launch physics as the real flight —
        otherwise the preview and the actual path drift apart."""
        pull_vec = (self.pos[0] - mouse_pos[0], self.pos[1] - mouse_pos[1])
        pull_len = math.hypot(*pull_vec)
        if pull_len < MIN_PULL:
            return None
        pull_len_clamped = min(pull_len, MAX_PULL)
        speed = MIN_LAUNCH_SPEED + (pull_len_clamped - MIN_PULL) / (MAX_PULL - MIN_PULL) * \
            (MAX_LAUNCH_SPEED - MIN_LAUNCH_SPEED)
        ux, uy = pull_vec[0] / pull_len, pull_vec[1] / pull_len
        power_frac = (speed - MIN_LAUNCH_SPEED) / (MAX_LAUNCH_SPEED - MIN_LAUNCH_SPEED)
        boost_duration = BOOST_RAMP_SECONDS * (0.75 + 0.5 * power_frac)
        fuel_budget = self.propellant_total * FUEL_LAUNCH_BURN_FRAC * (0.4 + 0.6 * power_frac)
        return {
            "dir": (ux, uy),
            "target_speed": speed,
            "boost_duration": boost_duration,
            "fuel_budget": fuel_budget,
        }

    def try_launch(self, mouse_pos):
        if self.fuel_remaining <= 0 or self.battery_remaining <= 0:
            return  # no propellant/power left — engines can't ignite
        params = self._launch_params(mouse_pos)
        if params is None:
            return  # too short a drag, cancel

        # --- gradual, realistic-feeling ignition: a small initial nudge,
        #     then thrust ramps the rocket up to commanded speed over a
        #     couple of seconds instead of an instant jump ---
        ux, uy = params["dir"]
        speed = params["target_speed"]
        self.launch_dir = (ux, uy)
        self.launch_target_speed = speed
        self.boost_duration = params["boost_duration"]
        self.boost_elapsed = 0.0
        self.vel = [ux * speed * LIFTOFF_SPEED_FRAC, uy * speed * LIFTOFF_SPEED_FRAC]

        # fuel for this burn is budgeted here but SPENT gradually across
        # the ramp in update() — not all at once
        self.launch_fuel_budget = params["fuel_budget"]

        self.state = "flight"
        if self.leg == "outbound":
            self.elapsed = 0.0
            self.trail = [tuple(self.pos)]
        else:
            self.trail.append(tuple(self.pos))

    # ------------------------------------------------------------------
    def simulate_step(self, pos, vel, dt, target_pos):
        """One physics step. Returns (new_pos, new_vel, dist_to_mars, status).
        Mars is the only body that exerts gravity (and the only thing you
        can crash into) — this is the gravity-assist / slingshot effect.
        Earth and the Target don't pull the rocket at all; reaching them
        is purely proximity-based (CAPTURE_RADIUS).

        FIX: Mars' gravity uses a *softened* distance
        (max(actual_dist, GRAVITY_SOFTENING)) in the 1/dist^2 term. This
        prevents the acceleration from spiking toward infinity when the
        rocket passes very close to Mars.
        """
        dx = self.mars_pos[0] - pos[0]
        dy = self.mars_pos[1] - pos[1]
        dist = math.hypot(dx, dy)
        if dist < CRASH_RADIUS:
            return pos, vel, dist, "crashed"

        vel = [vel[0], vel[1]]
        if dist < INFLUENCE_RADIUS:
            safe_dist = max(dist, GRAVITY_SOFTENING)
            accel = G_MARS / (safe_dist * safe_dist)
            vel[0] += accel * dx / safe_dist * dt
            vel[1] += accel * dy / safe_dist * dt

        pos = [pos[0] + vel[0] * dt, pos[1] + vel[1] * dt]
        tdx, tdy = target_pos[0] - pos[0], target_pos[1] - pos[1]
        if math.hypot(tdx, tdy) < CAPTURE_RADIUS:
            return pos, vel, dist, "success"
        if pos[0] < -250 or pos[0] > W + 250 or pos[1] < -250 or pos[1] > H + 250:
            return pos, vel, dist, "lost"
        return pos, vel, dist, "flying"

    # ------------------------------------------------------------------
    def preview_path(self, mouse_pos):
        """Ghost trajectory shown live while aiming (does not affect real state).
        Replays the exact same liftoff-nudge + gradual-thrust-ramp used by the
        real flight (see try_launch/update), so what you see during aiming is
        what you actually fly — not an instant-full-speed approximation."""
        params = self._launch_params(mouse_pos)
        if params is None:
            return [], None

        ux, uy = params["dir"]
        speed = params["target_speed"]
        boost_duration = params["boost_duration"]
        target_pos = self.target_pos if self.leg == "outbound" else EARTH_POS

        pos = list(self.pos)
        vel = [ux * speed * LIFTOFF_SPEED_FRAC, uy * speed * LIFTOFF_SPEED_FRAC]
        boost_elapsed = 0.0
        dt = 1 / 60
        thrust_accel = speed / boost_duration

        pts = []
        outcome = "flying"
        for i in range(7800):  # up to ~130 sim-seconds of flight
            if boost_elapsed < boost_duration:
                vel[0] += ux * thrust_accel * dt
                vel[1] += uy * thrust_accel * dt
                boost_elapsed += dt
            pos, vel, dist, outcome = self.simulate_step(pos, vel, dt, target_pos)
            if i % 4 == 0:
                pts.append((pos[0], pos[1]))
            if outcome in ("crashed", "success", "lost"):
                pts.append((pos[0], pos[1]))
                break
        return pts, outcome

    # ------------------------------------------------------------------
    def _manual_thrust_direction(self):
        """Reads WASD/Arrow keys and returns a normalized (dx, dy) thrust
        direction RELATIVE TO THE SHIP'S NOSE (self.heading_deg), or None
        if nothing is held. Up/W = forward (nose direction), Down/S = 
        backward, Left/A = strafe left, Right/D = strafe right."""
        keys = pygame.key.get_pressed()
        forward = int(keys[pygame.K_UP] or keys[pygame.K_w]) - int(keys[pygame.K_DOWN] or keys[pygame.K_s])
        strafe = int(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - int(keys[pygame.K_LEFT] or keys[pygame.K_a])
        if forward == 0 and strafe == 0:
            return None
        theta = math.radians(self.heading_deg)
        fx, fy = math.cos(theta), math.sin(theta)     # nose-forward unit vector
        rx, ry = -math.sin(theta), math.cos(theta)    # unit vector to the right of the nose
        dx = forward * fx + strafe * rx
        dy = forward * fy + strafe * ry
        dlen = math.hypot(dx, dy)
        if dlen < 1e-6:
            return None
        return (dx / dlen, dy / dlen)

    # ------------------------------------------------------------------
    def update(self, dt):
        if self.state != "flight":
            self.manual_active = False
            self.out_of_resources = self.fuel_remaining <= 0 or self.battery_remaining <= 0
            return

        self.elapsed += dt

        # can the ship actually fire any engines/thrusters this frame?
        resources_available = self.fuel_remaining > 0 and self.battery_remaining > 0
        self.out_of_resources = not resources_available

        # --- gradual launch thrust ramp (adds to velocity on top of
        #     whatever gravity is doing this frame) — only while powered ---
        if resources_available and self.boost_elapsed < self.boost_duration:
            thrust_accel = self.launch_target_speed / self.boost_duration
            self.vel[0] += self.launch_dir[0] * thrust_accel * dt
            self.vel[1] += self.launch_dir[1] * thrust_accel * dt
            burn = self.launch_fuel_budget * (dt / self.boost_duration)
            self.fuel_remaining = max(0.0, self.fuel_remaining - burn)
            self.boost_elapsed += dt

        # --- manual control: player-held thrust, expensive but powerful —
        #     unavailable once fuel or battery is exhausted ---
        manual_dir = self._manual_thrust_direction() if resources_available else None
        self.manual_active = manual_dir is not None
        if self.manual_active:
            self.vel[0] += manual_dir[0] * MANUAL_THRUST_ACCEL * dt
            self.vel[1] += manual_dir[1] * MANUAL_THRUST_ACCEL * dt

        target_pos = self.target_pos if self.leg == "outbound" else EARTH_POS
        self.pos, self.vel, dist_to_mars, status = self.simulate_step(self.pos, self.vel, dt, target_pos)
        self.speed_current = math.hypot(*self.vel)
        if self.speed_current > 1.0:
            self.heading_deg = math.degrees(math.atan2(self.vel[1], self.vel[0]))
        self.min_mars_dist_this_flight = min(self.min_mars_dist_this_flight, dist_to_mars)
        self.trail.append(tuple(self.pos))
        if len(self.trail) > 6000:
            self.trail.pop(0)

        # --- live battery drain (baseline systems + extra manual draw) ---
        mission_hours_elapsed = self.elapsed * HOURS_PER_SECOND
        used_wh = mission_hours_elapsed * self.avg_power_draw
        if self.manual_active:
            passive_wh_per_sec = self.avg_power_draw * HOURS_PER_SECOND
            self.manual_wh_used += passive_wh_per_sec * (MANUAL_BATTERY_DRAIN_MULT - 1.0) * dt
        self.battery_remaining = max(0.0, self.battery_total_wh - used_wh - self.manual_wh_used)

        # --- live fuel drain (continuous maneuvering / life support,
        #     multiplied way up while manual thrusters are firing) ---
        drain_mult = MANUAL_FUEL_DRAIN_MULT if self.manual_active else 1.0
        drain_rate = self.propellant_total * FUEL_DRAIN_OVER_60S / 60.0 * drain_mult
        self.fuel_remaining = max(0.0, self.fuel_remaining - drain_rate * dt)

        # --- random unpredictable events: these now actually knock the
        #     rocket off course and demand a manual correction ---
        if self.manual_alert is None and random.random() < EVENT_CHANCE_PER_FRAME:
            name, impact = random.choice(RANDOM_EVENTS)
            kick_angle = random.uniform(0, 2 * math.pi)
            kick_mag = EVENT_KICK_BASE + impact * EVENT_KICK_SCALE
            self.vel[0] += math.cos(kick_angle) * kick_mag
            self.vel[1] += math.sin(kick_angle) * kick_mag
            self.manual_alert = {"name": name, "timer": ALERT_GRACE_PERIOD, "correction": 0.0}
            self.event_log.append([f"! {name} — knocked off course!", 5.0])
            self.active_event_impacts.append([impact, 6.0])

        # --- resolve or expire the active alert ---
        if self.manual_alert is not None:
            if self.manual_active:
                self.manual_alert["correction"] += dt
                if self.manual_alert["correction"] >= MANUAL_CORRECTION_HOLD:
                    self.event_log.append([f"\u2713 {self.manual_alert['name']} corrected — back on course.", 3.5])
                    self.manual_alert = None
            if self.manual_alert is not None:
                self.manual_alert["timer"] -= dt
                if self.manual_alert["timer"] <= 0:
                    self.unresolved_penalty += UNRESOLVED_EVENT_RISK_PENALTY
                    self.event_log.append([f"\u2717 {self.manual_alert['name']} went uncorrected — risk up.", 4.0])
                    self.manual_alert = None

        # decay event log / impacts
        self.event_log = [[t, life - dt] for t, life in self.event_log if life - dt > 0]
        self.active_event_impacts = [[imp, life - dt] for imp, life in self.active_event_impacts
                                      if life - dt > 0]
        event_risk = sum(imp * (life / 6.0) for imp, life in self.active_event_impacts)

        # --- live risk score ---
        if dist_to_mars < INFLUENCE_RADIUS:
            proximity_risk = max(0.0, (INFLUENCE_RADIUS - dist_to_mars) /
                                  (INFLUENCE_RADIUS - CRASH_RADIUS)) * 0.4
        else:
            proximity_risk = 0.0
        duration_risk = min(self.elapsed / 90.0, 1.0) * 0.2
        self.risk_current = min(1.0, 0.15 + proximity_risk + duration_risk + event_risk + self.unresolved_penalty)

        # --- live cost ---
        fuel_used = self.propellant_total - self.fuel_remaining
        fuel_cost = fuel_used * 40
        ops_cost = self.elapsed * OPS_COST_PER_SECOND
        risk_premium = (self.hardware_cost + fuel_cost) * (self.risk_current * 0.3)
        self.cost_current = self.hardware_cost + fuel_cost + ops_cost + risk_premium

        # --- failure conditions ---
        if self.elapsed > MAX_MISSION_TIME:
            self.state = "lost"
            return

        if status != "flying":
            if status == "success" and self.leg == "outbound":
                # reached the target — now aim a slingshot back to Earth
                self.leg = "return"
                self.state = "aim"
                self.vel = [0.0, 0.0]
                self.event_log.append(["Target reached! Slingshot back to Earth to save money.", 6.0])
            elif status == "success" and self.leg == "return":
                # made it home — reusable hardware is recovered, saving cost
                recovery = self.hardware_cost * RECOVERY_FRACTION
                self.recovery_savings = recovery
                self.cost_current = max(0.0, self.cost_current - recovery)
                self.state = "returned"
            else:
                self.state = status

    # ------------------------------------------------------------------
    def draw_glow_circle(self, pos, radius, color, label=None):
        glow = pygame.Surface((radius * 5, radius * 5), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*color, 70), (radius * 2, radius * 2), int(radius * 1.8))
        self.screen.blit(glow, (pos[0] - radius * 2, pos[1] - radius * 2))
        pygame.draw.circle(self.screen, color, pos, radius)
        pygame.draw.circle(self.screen, (10, 10, 10), pos, radius, 2)
        if label:
            t = self.small_font.render(label, True, (235, 235, 235))
            self.screen.blit(t, (pos[0] - t.get_width() // 2, pos[1] + radius + 6))

    def draw_rocket(self, pos, heading_deg):
        size = 13
        theta = math.radians(heading_deg)
        tip = (pos[0] + size * math.cos(theta), pos[1] + size * math.sin(theta))
        left = (pos[0] + size * 0.55 * math.cos(theta + 2.6), pos[1] + size * 0.55 * math.sin(theta + 2.6))
        right = (pos[0] + size * 0.55 * math.cos(theta - 2.6), pos[1] + size * 0.55 * math.sin(theta - 2.6))
        pygame.draw.polygon(self.screen, (235, 235, 235), [tip, left, right])
        pygame.draw.polygon(self.screen, (20, 20, 20), [tip, left, right], 1)
        if self.state == "flight":
            flame_len = size * random.uniform(0.6, 1.1)
            back = (pos[0] - flame_len * math.cos(theta), pos[1] - flame_len * math.sin(theta))
            pygame.draw.line(self.screen, (255, 150, 30), pos, back, 3)

    # ------------------------------------------------------------------
    def draw(self):
        self.screen.fill((11, 16, 32))
        for sx, sy, r in self.stars:
            pygame.draw.circle(self.screen, (255, 255, 255), (int(sx), int(sy)), int(r))

        # Mars influence ring (subtle, for player understanding)
        ring = pygame.Surface((INFLUENCE_RADIUS * 2, INFLUENCE_RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(ring, (224, 122, 63, 25), (INFLUENCE_RADIUS, INFLUENCE_RADIUS), INFLUENCE_RADIUS)
        pygame.draw.circle(ring, (224, 122, 63, 60), (INFLUENCE_RADIUS, INFLUENCE_RADIUS), INFLUENCE_RADIUS, 1)
        self.screen.blit(ring, (self.mars_pos[0] - INFLUENCE_RADIUS, self.mars_pos[1] - INFLUENCE_RADIUS))

        earth_label = "Earth" if self.leg == "outbound" else "Earth (return home!)"
        self.draw_glow_circle(EARTH_POS, 24, (59, 130, 246), earth_label)
        self.draw_glow_circle(self.mars_pos, MARS_VISUAL_RADIUS, (224, 122, 63), "Mars")
        target_color = (192, 57, 43) if self.leg == "outbound" else (90, 90, 90)
        self.draw_glow_circle(self.target_pos, 26, target_color, "Target")

        # pulsing highlight on whichever body is the CURRENT destination
        pulse = 6 + 4 * math.sin(pygame.time.get_ticks() / 200.0)
        dest_pos = self.target_pos if self.leg == "outbound" else EARTH_POS
        if self.state == "aim":
            pygame.draw.circle(self.screen, (255, 255, 255), dest_pos, int(30 + pulse), 2)

        # flown trail
        if len(self.trail) > 1:
            pygame.draw.lines(self.screen, (127, 209, 255), False, self.trail, 2)

        # aiming: rubber band + ghost preview trajectory
        if self.state == "aim" and self.dragging and self.drag_mouse:
            pygame.draw.line(self.screen, (255, 255, 255), self.pos, self.drag_mouse, 2)
            pts, outcome = self.preview_path(self.drag_mouse)
            color = {"success": (0, 230, 120), "crashed": (230, 60, 60),
                     "lost": (150, 150, 150), "flying": (200, 200, 60)}.get(outcome, (200, 200, 60))
            for p in pts[::1]:
                pygame.draw.circle(self.screen, color, (int(p[0]), int(p[1])), 2)

        # rocket
        self.draw_rocket(self.pos, self.heading_deg)

        self.draw_hud()
        self.draw_alert_banner()
        self.draw_end_banner()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _manual_status_text(self):
        if self.out_of_resources:
            return "UNAVAILABLE (out of fuel/power — adrift)"
        if self.manual_active:
            return "ON (burning extra fuel/power)"
        return "off"

    def draw_hud(self):
        speed_kms = self.speed_current * 0.18  # stylized px/s -> "km/s" flavor conversion
        lines = [
            "MISSION STATS (live)",
            f"State:        {self.state.upper()}",
            f"Leg:          {self.leg.upper()}",
            f"Elapsed:      {self.elapsed:5.1f} s",
            f"Velocity:     {self.speed_current:6.1f} u/s  (~{speed_kms:4.1f} km/s)",
            f"Fuel:         {self.fuel_remaining:8,.0f} kg",
            f"Battery:      {self.battery_remaining:8,.0f} Wh",
            f"Risk score:   {self.risk_current:5.2f}",
            f"Est. cost:    ${self.cost_current:,.0f}",
            f"Manual ctrl:  {self._manual_status_text()}",
        ]
        if self.recovery_savings > 0:
            lines.append(f"Recovered:    ${self.recovery_savings:,.0f} saved!")

        panel_w = 320
        panel_h = 22 * len(lines) + 18
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(panel, (20, 26, 48, 215), (0, 0, panel_w, panel_h), border_radius=10)
        self.screen.blit(panel, (W - panel_w - 15, 15))
        for i, line in enumerate(lines):
            col = (255, 255, 255) if i == 0 else (225, 225, 225)
            if line.startswith("Recovered"):
                col = (0, 230, 120)
            if line.startswith("Manual ctrl") and self.manual_active:
                col = (255, 190, 60)
            if line.startswith("Manual ctrl") and self.out_of_resources:
                col = (230, 70, 70)
            t_surf = self.small_font.render(line, True, col)
            self.screen.blit(t_surf, (W - panel_w, 22 + i * 22))

        title = self.font.render("Flight Path Simulator — Slingshot Prototype", True, (255, 255, 255))
        self.screen.blit(title, (15, 15))

        if self.state == "aim":
            if self.out_of_resources:
                hint = "Out of fuel/power — engines are dead, the rocket can no longer maneuver"
            elif self.leg == "outbound":
                hint = "Click + drag the rocket back, then release to launch toward the Target"
            else:
                hint = "Target reached! Drag + release again to slingshot back to Earth and save money"
            surf = self.small_font.render(hint, True, (200, 200, 200))
            self.screen.blit(surf, (15, H - 30))
        elif self.state == "flight":
            if self.out_of_resources:
                hint = "Out of fuel/power — adrift on momentum and gravity alone, no thrust available"
            else:
                hint = "W/Up S/Down: forward/back thrust · A/Left D/Right: strafe — relative to the nose"
            surf = self.small_font.render(hint, True, (200, 200, 200))
            self.screen.blit(surf, (15, H - 30))

        # event log (top-left, fading)
        for i, (text, life) in enumerate(self.event_log[-4:]):
            alpha = max(0, min(255, int(life / 4.0 * 255)))
            surf = self.small_font.render(text, True, (255, 210, 60))
            surf.set_alpha(alpha)
            self.screen.blit(surf, (15, 50 + i * 20))

    def draw_alert_banner(self):
        if self.manual_alert is None:
            return
        flash_on = int(pygame.time.get_ticks() / 260) % 2 == 0
        color = (255, 90, 60) if flash_on else (255, 190, 60)
        timer_left = max(0.0, self.manual_alert["timer"])
        text = f"\u26a0 {self.manual_alert['name'].upper()} — HOLD WASD/ARROWS TO CORRECT ({timer_left:0.1f}s)"
        surf = self.font.render(text, True, color)
        box_w = surf.get_width() + 40
        box_h = surf.get_height() + 20
        box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        pygame.draw.rect(box, (30, 10, 10, 210), (0, 0, box_w, box_h), border_radius=10)
        pygame.draw.rect(box, color, (0, 0, box_w, box_h), 2, border_radius=10)
        self.screen.blit(box, (W // 2 - box_w // 2, 20))
        self.screen.blit(surf, (W // 2 - surf.get_width() // 2, 30))

    def draw_end_banner(self):
        messages = {
            "crashed": ("MISSION FAILED — Crashed into Mars", (230, 60, 60)),
            "lost": ("MISSION FAILED — Lost in space", (200, 200, 200)),
            "returned": ("MISSION SUCCESS — Round trip complete!", (0, 230, 120)),
        }
        if self.state in messages:
            text, color = messages[self.state]
            surf = self.big_font.render(text, True, color)
            self.screen.blit(surf, (W // 2 - surf.get_width() // 2, H // 2 - 60))
            if self.state == "returned":
                sub_text = f"Hardware recovered — saved ${self.recovery_savings:,.0f}. Press R to fly again."
            else:
                sub_text = "Press R to try again"
            sub = self.small_font.render(sub_text, True, (230, 230, 230))
            self.screen.blit(sub, (W // 2 - sub.get_width() // 2, H // 2 - 20))

    # ------------------------------------------------------------------
    async def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            running = self.handle_events()
            self.update(dt)
            self.draw()
            pygame.display.flip()
            await asyncio.sleep(0)  # required by pygbag — yields to the browser each frame
        pygame.quit()


if __name__ == "__main__":
    asyncio.run(SlingshotGame().run())
