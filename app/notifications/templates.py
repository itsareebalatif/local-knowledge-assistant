from __future__ import annotations

import html

_BRAND = "Personal Knowledge Engine"
_ACCENT = "#4f46e5"
_TEXT = "#1f2328"
_MUTED = "#6b7280"
_BORDER = "#e5e7eb"


def _button(label: str, url: str) -> str:
    return f"""
    <table role="presentation" cellspacing="0" cellpadding="0" style="margin: 28px 0;">
      <tr>
        <td style="border-radius: 8px; background: {_ACCENT};">
          <a href="{url}" target="_blank"
             style="display: inline-block; padding: 12px 28px; font-size: 15px; font-weight: 600;
                    color: #ffffff; text-decoration: none; border-radius: 8px; font-family: Arial, sans-serif;">
            {label}
          </a>
        </td>
      </tr>
    </table>
    """


def _layout(preheader: str, body_html: str) -> str:
    # preheader is hidden preview text shown next to the subject in inbox lists.
    return f"""
    <!DOCTYPE html>
    <html>
      <body style="margin: 0; padding: 0; background: #f3f4f6;
                   font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;">
        <span style="display: none; max-height: 0; overflow: hidden; opacity: 0;">{preheader}</span>
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background: #f3f4f6; padding: 32px 16px;">
          <tr>
            <td align="center">
              <table role="presentation" width="100%" style="max-width: 480px; background: #ffffff;
                     border-radius: 12px; overflow: hidden; border: 1px solid {_BORDER};">
                <tr>
                  <td style="padding: 28px 32px 0;">
                    <div style="font-size: 17px; font-weight: 700; color: {_TEXT};">{_BRAND}</div>
                  </td>
                </tr>
                <tr>
                  <td style="padding: 20px 32px 32px; color: {_TEXT}; font-size: 15px; line-height: 1.6;">
                    {body_html}
                  </td>
                </tr>
                <tr>
                  <td style="padding: 20px 32px; border-top: 1px solid {_BORDER};
                             color: {_MUTED}; font-size: 12px; line-height: 1.5;">
                    This is an automated message from {_BRAND}. If you weren't expecting it, you can ignore it.
                  </td>
                </tr>
              </table>
            </td>
          </tr>
        </table>
      </body>
    </html>
    """


def password_reset_email(full_name: str, reset_url: str) -> tuple[str, str]:
    subject = "Reset your Personal Knowledge Engine password"
    name = html.escape(full_name)
    body = f"""
    <p style="margin: 0 0 12px;">Hi {name},</p>
    <p style="margin: 0 0 12px;">
      We received a request to reset the password on your {_BRAND} account.
      Click the button below to choose a new one.
    </p>
    {_button("Reset password", reset_url)}
    <p style="margin: 0 0 12px; color: {_MUTED}; font-size: 13px;">
      This link expires in 30 minutes and can only be used once.
      If the button doesn't work, copy and paste this URL into your browser:<br>
      <a href="{reset_url}" style="color: {_ACCENT}; word-break: break-all;">{reset_url}</a>
    </p>
    <p style="margin: 20px 0 0; color: {_MUTED}; font-size: 13px;">
      Didn't request this? You can safely ignore this email — your password hasn't been changed.
    </p>
    """
    return subject, _layout("Reset your password — this link expires in 30 minutes.", body)


def welcome_email(full_name: str) -> tuple[str, str]:
    subject = f"Welcome to {_BRAND}"
    name = html.escape(full_name)
    body = f"""
    <p style="margin: 0 0 12px;">Hi {name},</p>
    <p style="margin: 0 0 12px;">Your account is ready.</p>
    <p style="margin: 0 0 12px;">
      Upload a document and start asking questions — answers are grounded and
      cited back to your own files, or refused outright when nothing in them
      actually supports an answer.
    </p>
    <p style="margin: 20px 0 0; color: {_MUTED}; font-size: 13px;">
      Everything you upload stays private to your account.
    </p>
    """
    return subject, _layout("Your account is ready — upload a document to get started.", body)
