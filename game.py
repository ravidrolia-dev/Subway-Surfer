"""Pygame implementation of the Gesture Runner endless runner."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List, Optional

import pygame

from config import (
    ACTION_JUMP,
    ACTION_LEFT_LANE,
    ACTION_MIDDLE_LANE,
    ACTION_NONE,
    ACTION_RIGHT_LANE,
    ACTION_SLIDE,
    ACTION_REPEAT_COOLDOWN_MS,
    COLOR_BACKGROUND,
    COLOR_BLOCK,
    COLOR_HIGH_OBSTACLE,
    COLOR_LANE_MARKER,
    COLOR_LOW_OBSTACLE,
    COLOR_MUTED_TEXT,
    COLOR_PLAYER,
    COLOR_PLAYER_DARK,
    COLOR_ROAD,
    COLOR_ROAD_EDGE,
    COLOR_SCORE,
    COLOR_SHADOW,
    COLOR_TEXT,
    GAME_TITLE,
    JUMP_DURATION_MS,
    JUMP_HEIGHT,
    LANE_ACTIONS,
    LANE_CENTERS,
    LANE_COUNT,
    OBSTACLE_CLEAR_BONUS,
    OBSTACLE_MAX_SPEED,
    OBSTACLE_SPEED_GAIN_PER_SCORE,
    OBSTACLE_START_SPEED,
    OBSTACLE_WIDTH,
    PLAYER_BASE_BOTTOM,
    PLAYER_HEIGHT,
    PLAYER_SLIDE_HEIGHT,
    PLAYER_START_LANE,
    PLAYER_WIDTH,
    ROAD_BOTTOM,
    ROAD_LEFT,
    ROAD_RIGHT,
    ROAD_TOP,
    SCORE_PER_SECOND,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SLIDE_DURATION_MS,
    SPAWN_INTERVAL_SCORE_FACTOR,
    SPAWN_MIN_INTERVAL_MS,
    SPAWN_START_INTERVAL_MS,
    TARGET_FPS,
)
from gesture_detector import GestureDetector, GestureResult


OBSTACLE_BLOCK = "block"
OBSTACLE_LOW = "low"
OBSTACLE_HIGH = "high"


@dataclass
class Player:
    """Player state: lane, jumping, and sliding."""

    lane: int = PLAYER_START_LANE
    jump_started_at: Optional[int] = None
    slide_started_at: Optional[int] = None

    def move_to_lane(self, lane: int) -> None:
        self.lane = max(0, min(LANE_COUNT - 1, lane))

    def start_jump(self, now_ms: int) -> bool:
        if not self.is_jumping(now_ms) and not self.is_sliding(now_ms):
            self.jump_started_at = now_ms
            return True
        return False

    def start_slide(self, now_ms: int) -> bool:
        if not self.is_jumping(now_ms) and not self.is_sliding(now_ms):
            self.slide_started_at = now_ms
            return True
        return False

    def update(self, now_ms: int) -> None:
        if self.jump_started_at is not None:
            if now_ms - self.jump_started_at >= JUMP_DURATION_MS:
                self.jump_started_at = None
        if self.slide_started_at is not None:
            if now_ms - self.slide_started_at >= SLIDE_DURATION_MS:
                self.slide_started_at = None

    def is_jumping(self, now_ms: int) -> bool:
        return (
            self.jump_started_at is not None
            and now_ms - self.jump_started_at < JUMP_DURATION_MS
        )

    def is_sliding(self, now_ms: int) -> bool:
        return (
            self.slide_started_at is not None
            and now_ms - self.slide_started_at < SLIDE_DURATION_MS
        )

    def rect(self, now_ms: int) -> pygame.Rect:
        height = PLAYER_SLIDE_HEIGHT if self.is_sliding(now_ms) else PLAYER_HEIGHT
        jump_offset = self._jump_offset(now_ms)
        rect = pygame.Rect(0, 0, PLAYER_WIDTH, height)
        rect.centerx = LANE_CENTERS[self.lane]
        rect.bottom = PLAYER_BASE_BOTTOM - jump_offset
        return rect

    def _jump_offset(self, now_ms: int) -> int:
        if not self.is_jumping(now_ms) or self.jump_started_at is None:
            return 0
        progress = (now_ms - self.jump_started_at) / JUMP_DURATION_MS
        return int(math.sin(progress * math.pi) * JUMP_HEIGHT)


@dataclass
class Obstacle:
    """Obstacle moving from the horizon toward the runner."""

    lane: int
    y: float
    kind: str
    passed: bool = False

    @property
    def height(self) -> int:
        if self.kind == OBSTACLE_LOW:
            return 42
        if self.kind == OBSTACLE_HIGH:
            return 96
        return 76

    @property
    def width(self) -> int:
        if self.kind == OBSTACLE_HIGH:
            return OBSTACLE_WIDTH + 14
        return OBSTACLE_WIDTH

    def rect(self) -> pygame.Rect:
        rect = pygame.Rect(0, int(self.y), self.width, self.height)
        rect.centerx = LANE_CENTERS[self.lane]
        return rect


class GestureRunnerGame:
    """Main game object for setup, loop, drawing, and cleanup."""

    def __init__(self, camera_index: int = 0) -> None:
        pygame.init()
        pygame.display.set_caption(GAME_TITLE)
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.small_font = pygame.font.Font(None, 22)
        self.font = pygame.font.Font(None, 30)
        self.big_font = pygame.font.Font(None, 72)
        self.mid_font = pygame.font.Font(None, 42)
        self.detector = GestureDetector(camera_index=camera_index)
        self.running = True
        self.game_over = False
        self.player = Player()
        self.obstacles: List[Obstacle] = []
        self.score = 0.0
        self.spawn_timer_ms = 0.0
        self.road_scroll = 0.0
        self.last_action_taken = ACTION_NONE
        self.last_action_source = "none"
        self.last_motion_action_time = -10_000
        self.last_console_log_time = 0
        self.gesture_result = GestureResult()

    def run(self) -> None:
        while self.running:
            dt_ms = self.clock.tick(TARGET_FPS)
            now_ms = pygame.time.get_ticks()
            keyboard_actions = self._handle_events()
            self.gesture_result = self.detector.detect()

            if not self.game_over:
                self._apply_keyboard_actions(keyboard_actions, now_ms)
                self._apply_gesture_action(now_ms)
                self._update(dt_ms, now_ms)

            self._draw(now_ms)
            pygame.display.flip()
            self._log_status(now_ms)

    def _handle_events(self) -> List[str]:
        actions: List[str] = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif self.game_over and event.key in (pygame.K_r, pygame.K_RETURN):
                    self.reset()
                elif not self.game_over:
                    action = self._keyboard_action(event.key)
                    if action:
                        actions.append(action)
        return actions

    def _keyboard_action(self, key: int) -> Optional[str]:
        if key in (pygame.K_a, pygame.K_LEFT):
            return ACTION_LEFT_LANE
        if key in (pygame.K_s, pygame.K_DOWN):
            return ACTION_MIDDLE_LANE
        if key in (pygame.K_d, pygame.K_RIGHT):
            return ACTION_RIGHT_LANE
        if key == pygame.K_SPACE:
            return ACTION_JUMP
        if key in (
            pygame.K_LCTRL,
            pygame.K_RCTRL,
            pygame.K_LSHIFT,
            pygame.K_RSHIFT,
        ):
            return ACTION_SLIDE
        return None

    def _apply_keyboard_actions(self, actions: List[str], now_ms: int) -> None:
        for action in actions:
            self._apply_action(action, now_ms, "keyboard")

    def _apply_gesture_action(self, now_ms: int) -> None:
        # Keyboard remains useful when no hand is detected or the camera is unavailable.
        if self.gesture_result.hand_detected and self.gesture_result.is_stable:
            self._apply_action(self.gesture_result.action, now_ms, "gesture")

    def _apply_action(self, action: str, now_ms: int, source: str) -> None:
        if action in LANE_ACTIONS:
            self.player.move_to_lane(LANE_ACTIONS[action])
            self._record_action(action, source)
            return

        if now_ms - self.last_motion_action_time < ACTION_REPEAT_COOLDOWN_MS:
            return

        started = False
        if action == ACTION_JUMP:
            started = self.player.start_jump(now_ms)
        elif action == ACTION_SLIDE:
            started = self.player.start_slide(now_ms)

        if started:
            self.last_motion_action_time = now_ms
            self._record_action(action, source)

    def _record_action(self, action: str, source: str) -> None:
        self.last_action_taken = action
        self.last_action_source = source

    def _update(self, dt_ms: int, now_ms: int) -> None:
        dt_seconds = dt_ms / 1000.0
        self.score += SCORE_PER_SECOND * dt_seconds
        self.player.update(now_ms)
        speed = self._current_speed()
        self.road_scroll = (self.road_scroll + speed * dt_seconds) % 76
        self._spawn_obstacles(dt_ms)
        self._move_obstacles(speed, dt_seconds)
        self._check_collisions(now_ms)

    def _current_speed(self) -> float:
        return min(
            OBSTACLE_MAX_SPEED,
            OBSTACLE_START_SPEED + self.score * OBSTACLE_SPEED_GAIN_PER_SCORE,
        )

    def _current_spawn_interval(self) -> float:
        return max(
            SPAWN_MIN_INTERVAL_MS,
            SPAWN_START_INTERVAL_MS - self.score * SPAWN_INTERVAL_SCORE_FACTOR,
        )

    def _spawn_obstacles(self, dt_ms: int) -> None:
        self.spawn_timer_ms += dt_ms
        if self.spawn_timer_ms < self._current_spawn_interval():
            return

        self.spawn_timer_ms = 0.0
        lane = random.randrange(LANE_COUNT)
        kind = random.choices(
            [OBSTACLE_BLOCK, OBSTACLE_LOW, OBSTACLE_HIGH],
            weights=[0.46, 0.29, 0.25],
            k=1,
        )[0]
        self.obstacles.append(Obstacle(lane=lane, y=-110, kind=kind))

    def _move_obstacles(self, speed: float, dt_seconds: float) -> None:
        for obstacle in self.obstacles:
            obstacle.y += speed * dt_seconds
            if not obstacle.passed and obstacle.y > PLAYER_BASE_BOTTOM:
                obstacle.passed = True
                self.score += OBSTACLE_CLEAR_BONUS

        self.obstacles = [
            obstacle for obstacle in self.obstacles if obstacle.y < SCREEN_HEIGHT + 120
        ]

    def _check_collisions(self, now_ms: int) -> None:
        player_rect = self.player.rect(now_ms)
        for obstacle in self.obstacles:
            if not player_rect.colliderect(obstacle.rect()):
                continue

            if obstacle.kind == OBSTACLE_LOW and self.player.is_jumping(now_ms):
                continue
            if obstacle.kind == OBSTACLE_HIGH and self.player.is_sliding(now_ms):
                continue

            self.game_over = True
            self._record_action("collision", "game")
            print(
                "[Gesture Runner] GAME OVER "
                f"gesture={self.gesture_result.gesture} "
                f"action={self.last_action_taken} score={int(self.score)}"
            )
            break

    def _draw(self, now_ms: int) -> None:
        self.screen.fill(COLOR_BACKGROUND)
        self._draw_road()
        self._draw_obstacles()
        self._draw_player(now_ms)
        self._draw_camera_preview()
        self._draw_hud()
        if self.game_over:
            self._draw_game_over()

    def _draw_road(self) -> None:
        road_rect = pygame.Rect(ROAD_LEFT, ROAD_TOP, ROAD_RIGHT - ROAD_LEFT, ROAD_BOTTOM)
        pygame.draw.rect(self.screen, COLOR_ROAD_EDGE, road_rect.inflate(34, 0))
        pygame.draw.rect(self.screen, COLOR_ROAD, road_rect)

        for lane_index in range(1, LANE_COUNT):
            x = ROAD_LEFT + lane_index * ((ROAD_RIGHT - ROAD_LEFT) // LANE_COUNT)
            dash_y = -76 + int(self.road_scroll)
            while dash_y < SCREEN_HEIGHT:
                pygame.draw.rect(
                    self.screen,
                    COLOR_LANE_MARKER,
                    pygame.Rect(x - 3, dash_y, 6, 42),
                    border_radius=3,
                )
                dash_y += 76

    def _draw_obstacles(self) -> None:
        for obstacle in self.obstacles:
            rect = obstacle.rect()
            shadow = rect.copy()
            shadow.y += 7
            pygame.draw.rect(self.screen, COLOR_SHADOW, shadow, border_radius=6)

            if obstacle.kind == OBSTACLE_LOW:
                pygame.draw.rect(self.screen, COLOR_LOW_OBSTACLE, rect, border_radius=7)
                pygame.draw.rect(
                    self.screen,
                    (122, 74, 20),
                    rect.inflate(-16, -16),
                    width=4,
                    border_radius=4,
                )
            elif obstacle.kind == OBSTACLE_HIGH:
                pygame.draw.rect(self.screen, COLOR_HIGH_OBSTACLE, rect, border_radius=7)
                inner = rect.inflate(-20, -58)
                inner.centery = rect.centery
                pygame.draw.rect(self.screen, (232, 224, 255), inner, border_radius=4)
            else:
                pygame.draw.rect(self.screen, COLOR_BLOCK, rect, border_radius=7)
                pygame.draw.rect(
                    self.screen,
                    (255, 210, 210),
                    rect.inflate(-18, -18),
                    width=3,
                    border_radius=4,
                )

    def _draw_player(self, now_ms: int) -> None:
        rect = self.player.rect(now_ms)
        shadow = pygame.Rect(0, 0, PLAYER_WIDTH + 20, 14)
        shadow.centerx = rect.centerx
        shadow.centery = PLAYER_BASE_BOTTOM + 7
        pygame.draw.ellipse(self.screen, COLOR_SHADOW, shadow)
        pygame.draw.rect(self.screen, COLOR_PLAYER, rect, border_radius=10)

        head_radius = 19 if not self.player.is_sliding(now_ms) else 14
        head_center = (rect.centerx, rect.top + head_radius + 5)
        pygame.draw.circle(self.screen, COLOR_PLAYER_DARK, head_center, head_radius)

        # Small white marks make the simple runner shape easier to track.
        pygame.draw.circle(
            self.screen, COLOR_TEXT, (head_center[0] - 7, head_center[1] - 3), 3
        )
        pygame.draw.circle(
            self.screen, COLOR_TEXT, (head_center[0] + 7, head_center[1] - 3), 3
        )

    def _draw_hud(self) -> None:
        fps = self.clock.get_fps()
        score_text = self.font.render(f"Score: {int(self.score)}", True, COLOR_SCORE)
        fps_text = self.font.render(f"FPS: {fps:.1f}", True, COLOR_TEXT)
        gesture_text = self.font.render(
            f"Gesture: {self.gesture_result.gesture}", True, COLOR_TEXT
        )
        raw_text = self.font.render(
            f"Raw: {self.gesture_result.raw_gesture}", True, COLOR_MUTED_TEXT
        )
        action_text = self.font.render(
            f"Action: {self.last_action_taken} ({self.last_action_source})",
            True,
            COLOR_MUTED_TEXT,
        )
        camera_note = ""
        if self.gesture_result.message:
            camera_note = self.gesture_result.message
        elif self.gesture_result.finger_count is not None:
            camera_note = f"Fingers: {self.gesture_result.finger_count}"
        note_text = self.font.render(camera_note, True, COLOR_MUTED_TEXT)

        self.screen.blit(score_text, (24, 20))
        self.screen.blit(fps_text, (24, 52))
        self.screen.blit(gesture_text, (24, SCREEN_HEIGHT - 126))
        self.screen.blit(raw_text, (24, SCREEN_HEIGHT - 94))
        self.screen.blit(action_text, (24, SCREEN_HEIGHT - 62))
        self.screen.blit(note_text, (24, SCREEN_HEIGHT - 30))

    def _draw_camera_preview(self) -> None:
        panel = pygame.Rect(SCREEN_WIDTH - 236, 18, 220, 196)
        pygame.draw.rect(self.screen, (30, 34, 44), panel, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_MUTED_TEXT, panel, width=1, border_radius=6)

        title = self.small_font.render(
            f"Camera {self.detector.camera_index} / {self.gesture_result.backend}",
            True,
            COLOR_TEXT,
        )
        self.screen.blit(title, (panel.x + 8, panel.y + 6))

        frame = self.detector.get_latest_rgb_frame()
        frame_rect = pygame.Rect(panel.x + 8, panel.y + 30, 204, 153)
        if frame is None:
            message = self.small_font.render("No camera frame", True, COLOR_MUTED_TEXT)
            self.screen.blit(message, message.get_rect(center=frame_rect.center))
            return

        try:
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            surface = pygame.transform.smoothscale(surface, frame_rect.size)
            self.screen.blit(surface, frame_rect)
        except Exception:
            message = self.small_font.render("Preview unavailable", True, COLOR_MUTED_TEXT)
            self.screen.blit(message, message.get_rect(center=frame_rect.center))

    def _draw_game_over(self) -> None:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 172))
        self.screen.blit(overlay, (0, 0))

        title = self.big_font.render("GAME OVER", True, COLOR_TEXT)
        score = self.mid_font.render(f"Final Score: {int(self.score)}", True, COLOR_SCORE)
        restart = self.font.render("Press R or Enter to restart", True, COLOR_TEXT)
        quit_text = self.font.render("Press Esc to quit", True, COLOR_MUTED_TEXT)

        self.screen.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, 230)))
        self.screen.blit(score, score.get_rect(center=(SCREEN_WIDTH // 2, 302)))
        self.screen.blit(restart, restart.get_rect(center=(SCREEN_WIDTH // 2, 370)))
        self.screen.blit(quit_text, quit_text.get_rect(center=(SCREEN_WIDTH // 2, 406)))

    def _log_status(self, now_ms: int) -> None:
        if now_ms - self.last_console_log_time < 1000:
            return
        self.last_console_log_time = now_ms
        print(
            "[Gesture Runner] "
            f"gesture={self.gesture_result.gesture} "
            f"raw={self.gesture_result.raw_gesture} "
            f"hand={self.gesture_result.hand_detected} "
            f"stable={self.gesture_result.is_stable} "
            f"action={self.last_action_taken} "
            f"source={self.last_action_source} "
            f"fps={self.clock.get_fps():.1f} "
            f"score={int(self.score)}"
        )

    def reset(self) -> None:
        self.game_over = False
        self.player = Player()
        self.obstacles.clear()
        self.score = 0.0
        self.spawn_timer_ms = 0.0
        self.road_scroll = 0.0
        self.last_action_taken = ACTION_NONE
        self.last_action_source = "none"
        self.last_motion_action_time = -10_000

    def close(self) -> None:
        self.detector.close()
        pygame.quit()
